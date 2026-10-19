// Common local-search engine for multi-entry graph ANN (stage 1 of the design).
//
// One CTA per (query, branch). Each branch owns an independent best-L queue, an
// independent open-addressing visited table, and its own counters. The queue is
// kept sorted ascending by the total order (distance, id), and the `expanded`
// flag travels with its entry through every merge. A branch stops when its
// expansion budget is spent or when no retained candidate is left unexpanded --
// never merely because a round failed to change the queue.
//
// All state lives in caller-owned tensors, so `advance` is resumable: running
// delta=a then delta=b must equal running delta=a+b from the same init. That
// equality is what stage 2's probe/resume depends on, and it is a test here.
//
// A branch may be seeded with V entries rather than one. They share the queue,
// the visited table and the budget -- which is the point: independent branches
// re-evaluate each other's nodes and each gets a budget too small to go deep,
// and measurement says that costs more than the extra entries are worth. Each
// queue entry carries the lineage of the seed it descends from, so expansion
// can be taken in turn across lineages (POLICY_ROUND_ROBIN) instead of always
// going to the globally nearest candidate (POLICY_BEST). Taking turns is what
// stops the nearest seed's basin from starving the others, which is the whole
// reason for having several.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK   128
#define MAXL    256   // L, the queue width, may not exceed this
#define MAXD    128   // D, the graph out-degree, may not exceed this (== BLOCK)
#define NCTR    6     // [0] expansions, [1] distance evals, [2] hash failures,
                      // [3] exhausted, [4] cache hits, [5] cache slots claimed
#define CPROBE  32    // bounded probe into the shared cache; a miss just recomputes

// Exclusive mode: the per-query cache doubles as a claim. A node another branch
// has already taken is skipped outright rather than entered with its cached
// distance, so branches cannot retread each other's ground and are pushed into
// territory nobody has. Measured reason: among a query's 64 nearest candidate
// entries, 94% of pairs end up recovering overlapping fragments -- the entries
// funnel into one basin, so diversity has to come from the trajectories rather
// than from where they start.
static const int POLICY_BEST = 0;          // always the nearest unexpanded candidate
static const int POLICY_ROUND_ROBIN = 1;   // nearest unexpanded of lineage (step % V)

// Stopping. A lineage is retired when its own best unexpanded candidate lies
// beyond (1 + stop_eps) times the current k-th best distance: nothing it can
// still reach will beat the answer, so its turns are waste. The slack is what
// keeps a lineage that is *travelling* -- crossing worse ground on its way to a
// fragment it has not reached yet -- from being retired on arrival's doorstep,
// and measurement is what sets it: at eps=0.25 the recall is the same as running
// the queue to exhaustion. A negative stop_eps disables the rule entirely.
#define NO_EARLY_STOP (-1.0f)

// The bound is the best k-th distance any of this query's CTAs has reached. Every
// branch tests its own frontier against it rather than against its own queue, so
// a branch that landed somewhere the answer has already passed stops at once
// instead of mining a region that cannot contribute. Distances are non-negative,
// so their bit patterns order the same way as the floats and a plain integer
// atomicMin is a float atomicMin.
__device__ __forceinline__ void bound_min(float* bound, float v) {
    atomicMin((int*)bound, __float_as_int(v));
}

static const int METRIC_IP = 0;   // distance = 1 - dot, for L2-normalised vectors
static const int METRIC_L2 = 1;   // distance = ||q - x||^2

// Total order on (distance, id). Ids are unique, so this is a strict total order
// and every tie-break is reproducible on CPU.
__device__ __forceinline__ bool lt(float da, int ia, float db, int ib) {
    return (da < db) || (da == db && ia < ib);
}

// Open addressing, linear probing, slot holds id+1 (0 = empty).
// Returns 1 if this call was the one that inserted the key.
// A full table is reported through `fail` and never silently swallowed.
__device__ __forceinline__ int hash_insert(int* table, int H, int key, int* fail) {
    unsigned mask = (unsigned)(H - 1);
    unsigned pos = (2654435761u * (unsigned)key) & mask;
    for (int probe = 0; probe < H; ++probe) {
        int cur = table[pos];
        if (cur == 0) {
            int old = atomicCAS(&table[pos], 0, key + 1);
            if (old == 0)       return 1;
            if (old == key + 1) return 0;
        } else if (cur == key + 1) {
            return 0;
        }
        pos = (pos + 1) & mask;
    }
    atomicAdd(fail, 1);
    return 0;
}

// Lanes of one team occupy a contiguous sub-warp segment. Every shuffle below
// names exactly that segment: teams diverge from each other (a team may skip an
// already-visited neighbour), so a full-warp mask would be a lie.
template <int TEAM>
__device__ __forceinline__ unsigned team_mask(int tid) {
    if (TEAM == 32) return 0xffffffffu;
    return (((1u << TEAM) - 1u) << (((tid % 32) / TEAM) * TEAM));
}

// The search reads one full vector per distance and does one multiply-add per
// element, so it is bound by how many bytes a candidate costs, not by the
// arithmetic -- measured at 1.76 TB/s on a part whose HBM is about 1.8. Storing
// the corpus in half therefore buys close to the factor its size says it should.
// The query stays in fp32 in shared memory and accumulation stays in fp32: only
// the bytes crossing the bus shrink.
// A lane loads 16 bytes at a time -- 4 floats or 8 halves -- so a 32-lane team
// pulls 512 contiguous bytes per step. Element-at-a-time half loads move two
// bytes per lane per instruction and leave most of the bus idle, which is the
// difference between 1.3 and 1.8 TB/s on this part.
template <typename T> struct Vec;
template <> struct Vec<float>  { static const int N = 4; };
template <> struct Vec<__half> { static const int N = 8; };

template <typename T>
__device__ __forceinline__ void load_vec(const T* p, float* out);

template <>
__device__ __forceinline__ void load_vec<float>(const float* p, float* out) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

template <>
__device__ __forceinline__ void load_vec<__half>(const __half* p, float* out) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        out[2 * i] = f.x; out[2 * i + 1] = f.y;
    }
}

// A distance belongs to a (query, node) pair, not to a branch, so when a query is
// searched by many branches at once they should compute each one exactly once.
// The per-branch visited table still exists and still does what it did -- keep a
// node out of the same branch's queue twice -- but the arithmetic now goes
// through a table shared by every branch of the query.
//
// Nothing about the result changes: a recomputed distance is bitwise equal to
// the cached one, because the inputs and the reduction order are the same. So a
// miss, a lost race or a full table costs work and never correctness, and the
// probe is bounded for the same reason -- past CPROBE steps it is cheaper to
// recompute than to keep looking.
//
// The table lives in L2: accesses are volatile so they bypass the per-SM L1,
// which is not coherent between the CTAs sharing this query.
__device__ __forceinline__ int cache_slot(int* table, int H, int key, int* claimed) {
    unsigned mask = (unsigned)(H - 1);
    unsigned pos = (2654435761u * (unsigned)key) & mask;
    for (int probe = 0; probe < CPROBE; ++probe) {
        int cur = ((volatile int*)table)[pos];
        if (cur == key + 1) return (int)pos;
        if (cur == 0) {
            int old = atomicCAS(&table[pos], 0, key + 1);
            if (old == 0)       { *claimed = 1; return (int)pos; }
            if (old == key + 1) return (int)pos;
        }
        pos = (pos + 1) & mask;
    }
    return -1;                       // give up and compute it; never a failure
}

template <int TEAM, typename XT>
__device__ __forceinline__ float team_dist(const float* __restrict__ q,
                                           const XT* __restrict__ x,
                                           int dim, int metric, int lane, unsigned tmask) {
    constexpr int V = Vec<XT>::N;
    float acc = 0.f, buf[V];
    for (int base = lane * V; base < dim; base += TEAM * V) {
        load_vec<XT>(x + base, buf);
        #pragma unroll
        for (int i = 0; i < V; ++i) {
            if (base + i >= dim) break;
            if (metric == METRIC_IP) acc += q[base + i] * buf[i];
            else { const float df = q[base + i] - buf[i]; acc += df * df; }
        }
    }
    for (int off = TEAM / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(tmask, acc, off, TEAM);
    return (metric == METRIC_IP) ? (1.f - acc) : acc;   // meaningful in lane 0
}

// Sort the first `n` candidate slots by (distance, id); never-filled slots carry
// FLT_MAX / -1 and land at the end. `n` is the smallest power of two covering the
// graph's out-degree rather than MAXD: at D 64 that is 21 compare-exchange
// stages instead of 28, and 21 barriers instead of 28, on every expansion. In a
// wide fan-out -- many shallow branches -- that per-expansion fixed cost is what
// the kernel is actually spending its time on.
__device__ __forceinline__ void bitonic_sort(int* c_id, float* c_d,
                                             unsigned char* c_l, int tid, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int s = k >> 1; s > 0; s >>= 1) {
            const int i = tid, ixj = tid ^ s;
            if (tid < n && ixj > i) {
                const bool asc = ((i & k) == 0);
                const bool need = asc ? lt(c_d[ixj], c_id[ixj], c_d[i], c_id[i])
                                      : lt(c_d[i], c_id[i], c_d[ixj], c_id[ixj]);
                if (need) {
                    float td = c_d[i]; c_d[i] = c_d[ixj]; c_d[ixj] = td;
                    int   ti = c_id[i]; c_id[i] = c_id[ixj]; c_id[ixj] = ti;
                    unsigned char tl = c_l[i]; c_l[i] = c_l[ixj]; c_l[ixj] = tl;
                }
            }
            __syncthreads();
        }
    }
}

template <int TEAM, typename XT>
__global__ void init_kernel(const XT* __restrict__ X,
                            const float* __restrict__ Qv,
                            const int*   __restrict__ seeds,
                            int* ids, float* dist, unsigned char* expd,
                            unsigned char* lin, int* visited, int* counters,
                            int dim, int J, int V, int L, int H, int metric) {
    const int j = blockIdx.x, q = blockIdx.y;
    const int tid = threadIdx.x, team = tid / TEAM, lane = tid % TEAM, nteams = BLOCK / TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    extern __shared__ float sq[];                    // dim floats: the query vector

    __shared__ int   c_id[MAXD];  __shared__ float c_d[MAXD];
    __shared__ unsigned char c_l[MAXD];

    const size_t bid = (size_t)q * J + j;
    int*  vis = visited  + bid * H;
    int*  ctr = counters + bid * NCTR;
    int*  qid = ids      + bid * L;
    float* qd = dist     + bid * L;
    unsigned char* qe = expd + bid * L;
    unsigned char* ql = lin  + bid * L;

    for (int t = tid; t < dim; t += BLOCK) sq[t] = Qv[(size_t)q * dim + t];
    for (int t = tid; t < H; t += BLOCK) vis[t] = 0;
    int nsort = 2; while (nsort < V && nsort < MAXD) nsort <<= 1;
    for (int i = tid; i < nsort; i += BLOCK) { c_id[i] = -1; c_d[i] = FLT_MAX; c_l[i] = 0; }
    for (int i = tid; i < NCTR; i += BLOCK) ctr[i] = 0;
    __syncthreads();

    // Every seed is evaluated and inserted; a seed repeated inside one branch is
    // kept once, by the same visited table that dedups the search itself.
    for (int base = 0; base < V; base += nteams) {
        const int si = base + team;
        if (si < V) {
            const int v = seeds[(bid * (size_t)V) + si];
            if (v >= 0) {
                int isnew = 0;
                if (lane == 0) isnew = hash_insert(vis, H, v, &ctr[2]);
                isnew = __shfl_sync(tmask, isnew, 0, TEAM);
                if (isnew) {
                    float d = team_dist<TEAM, XT>(sq, X + (size_t)v * dim, dim, metric, lane, tmask);
                    if (lane == 0) {
                        c_id[si] = v; c_d[si] = d; c_l[si] = (unsigned char)si;
                        atomicAdd(&ctr[1], 1);
                    }
                }
            }
        }
        __syncthreads();
    }

    bitonic_sort(c_id, c_d, c_l, tid, nsort);

    for (int i = tid; i < L; i += BLOCK) {
        const bool ok = (i < nsort) && (c_id[i] >= 0);
        qid[i] = ok ? c_id[i] : -1;
        qd[i]  = ok ? c_d[i]  : FLT_MAX;
        qe[i]  = ok ? 0 : 1;
        ql[i]  = ok ? c_l[i] : 0;
    }
}

template <int TEAM, typename XT>
__global__ void advance_kernel(const XT* __restrict__ X,
                               const int*   __restrict__ G,
                               const float* __restrict__ Qv,
                               int* ids, float* dist, unsigned char* expd,
                               unsigned char* lin, int* visited, int* counters,
                               int dim, int D, int J, int L, int H,
                               int metric, int delta, int max_expansions,
                               int policy, int V, float stop_eps, int kstop,
                               int* cache_id, float* cache_d, int Hc, float* bound,
                               int exclusive) {
    const int j = blockIdx.x, q = blockIdx.y;
    const int tid = threadIdx.x;
    const int team = tid / TEAM, lane = tid % TEAM, nteams = BLOCK / TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    extern __shared__ float sq[];

    __shared__ int   s_id[MAXL];  __shared__ float s_d[MAXL];
    __shared__ unsigned char s_e[MAXL], s_l[MAXL];
    __shared__ int   n_id[MAXL];  __shared__ float n_d[MAXL];
    __shared__ unsigned char n_e[MAXL], n_l[MAXL];
    __shared__ int   c_id[MAXD]; __shared__ float c_d[MAXD];
    __shared__ unsigned char c_l[MAXD];
    __shared__ int   s_slot, s_p, s_plin;
    // 256 lineage bits, because lineage is a uint8 and V may be up to 256. This
    // was one 32-bit word, which silently shifted out of range for V > 32 -- the
    // measurements at V = 32 sat exactly on that boundary.
    __shared__ unsigned s_retired[8];

    const size_t bid = (size_t)q * J + j;
    int*  vis = visited  + bid * H;
    int*  ctr = counters + bid * NCTR;
    int*  qid = ids      + bid * L;
    float* qd = dist     + bid * L;
    unsigned char* qe = expd + bid * L;
    unsigned char* ql = lin  + bid * L;
    int*   cid = (Hc > 0) ? cache_id + (size_t)q * Hc : nullptr;
    float* cdd = (Hc > 0) ? cache_d  + (size_t)q * Hc : nullptr;

    for (int t = tid; t < dim; t += BLOCK) sq[t] = Qv[(size_t)q * dim + t];
    for (int i = tid; i < L; i += BLOCK) {
        s_id[i] = qid[i]; s_d[i] = qd[i]; s_e[i] = qe[i]; s_l[i] = ql[i];
    }
    int nsort = 2; while (nsort < D && nsort < MAXD) nsort <<= 1;
    // Retirement is re-derived every call rather than stored: it is a function
    // of the state, it is monotone, and a re-check costs no expansion.
    for (int i = tid; i < 8; i += BLOCK) s_retired[i] = 0u;
    __syncthreads();

    for (int step = 0; step < delta; ++step) {
        if (ctr[0] >= max_expansions) break;          // uniform: ctr[0] only moves in tid 0
        if (tid == 0) {
            s_slot = -1;
            const bool early = (stop_eps >= 0.f);
            float radius = (kstop > 0 && kstop <= L && s_id[kstop - 1] >= 0)
                           ? s_d[kstop - 1] : FLT_MAX;
            if (bound) {                      // what the whole query knows, not this branch
                const float gb = ((volatile float*)bound)[q];
                if (gb < radius) radius = gb;
            }
            const float limit = (radius == FLT_MAX) ? FLT_MAX : radius * (1.f + stop_eps);

            if (policy == POLICY_ROUND_ROBIN && V > 1) {
                // Each lineage gets its turn, so the nearest seed's basin cannot
                // starve the others -- and a lineage that is done, or that can no
                // longer reach anything good, gives its turns back instead of
                // spending them.
                for (int turn = 0; turn < V && s_slot < 0; ++turn) {
                    const int want = (ctr[0] + turn) % V;
                    if (s_retired[want >> 5] & (1u << (want & 31))) continue;
                    int found = -1;
                    for (int i = 0; i < L; ++i)
                        if (s_id[i] >= 0 && s_e[i] == 0 && s_l[i] == want) { found = i; break; }
                    if (found < 0) { s_retired[want >> 5] |= (1u << (want & 31)); continue; }
                    if (early && s_d[found] > limit) {
                        s_retired[want >> 5] |= (1u << (want & 31)); continue;
                    }
                    s_slot = found;
                }
                if (s_slot < 0 && !early) {          // legacy fallback: global best
                    for (int i = 0; i < L; ++i)
                        if (s_id[i] >= 0 && s_e[i] == 0) { s_slot = i; break; }
                }
            } else {
                for (int i = 0; i < L; ++i)
                    if (s_id[i] >= 0 && s_e[i] == 0) {
                        if (!early || s_d[i] <= limit) s_slot = i;
                        break;
                    }
            }
            if (s_slot >= 0) {
                s_e[s_slot] = 1; s_p = s_id[s_slot]; s_plin = s_l[s_slot]; ctr[0] += 1;
            } else {
                ctr[3] = 1;
            }
        }
        __syncthreads();
        if (s_slot < 0) break;
        const int p = s_p;
        const unsigned char plin = (unsigned char)s_plin;

        for (int i = tid; i < nsort; i += BLOCK) { c_id[i] = -1; c_d[i] = FLT_MAX; c_l[i] = 0; }
        __syncthreads();

        // Expand p. Candidate i keeps the fixed slot i of G[p], so which lane did
        // the work never changes the result, and it inherits p's lineage.
        for (int base = 0; base < D; base += nteams) {
            const int nidx = base + team;
            if (nidx < D) {
                const int v = G[(size_t)p * D + nidx];
                if (v >= 0) {
                    int isnew = 0;
                    if (lane == 0) isnew = hash_insert(vis, H, v, &ctr[2]);
                    isnew = __shfl_sync(tmask, isnew, 0, TEAM);
                    if (isnew) {
                        int slot = -1, claimed = 0;
                        float d = FLT_MAX;
                        if (Hc > 0) {
                            if (lane == 0) {
                                slot = cache_slot(cid, Hc, v, &claimed);
                                d = (slot >= 0 && !claimed) ? ((volatile float*)cdd)[slot]
                                                            : FLT_MAX;
                            }
                            slot = __shfl_sync(tmask, slot, 0, TEAM);
                            d = __shfl_sync(tmask, d, 0, TEAM);
                        }
                        if (exclusive && slot >= 0 && !claimed && d != FLT_MAX) {
                            // somebody else owns this node; leave it to them
                            if (lane == 0) atomicAdd(&ctr[4], 1);
                        } else if (d == FLT_MAX) {               // not cached yet
                            d = team_dist<TEAM, XT>(sq, X + (size_t)v * dim, dim, metric, lane, tmask);
                            if (lane == 0) {
                                if (slot >= 0) { ((volatile float*)cdd)[slot] = d;
                                                 atomicAdd(&ctr[5], 1); }
                                atomicAdd(&ctr[1], 1);
                            }
                        } else if (lane == 0) {
                            atomicAdd(&ctr[4], 1);
                        }
                        const bool taken = exclusive && slot >= 0 && !claimed && d != FLT_MAX;
                        if (lane == 0 && !taken) {
                            c_id[nidx] = v; c_d[nidx] = d; c_l[nidx] = plin;
                        }
                    }
                }
            }
            __syncthreads();
        }

        bitonic_sort(c_id, c_d, c_l, tid, nsort);

        // Two sorted runs -> best L, one output slot per thread.
        //
        // The serial two-pointer merge this replaces ran up to L + D steps in
        // thread 0 with the other 127 idle, every expansion. Merge-path instead:
        // output slot o is fed by exactly i elements of the queue and o - i of
        // the candidates, and that i is found by a binary search of log(L) steps
        // that each thread can run on its own. Same output, same tie-break.
        for (int o = tid; o < L; o += BLOCK) {
            int lo = max(0, o - nsort), hi = min(o, L);
            while (lo < hi) {
                const int i = (lo + hi) >> 1, j = o - i;
                if (j > 0 && !lt(c_d[j - 1], c_id[j - 1], s_d[i], s_id[i])) lo = i + 1;
                else                                                       hi = i;
            }
            const int i = lo, j = o - i;
            const bool takeA = (j >= nsort) ||
                               (i < L && lt(s_d[i], s_id[i], c_d[j], c_id[j]));
            if (takeA) { n_id[o] = s_id[i]; n_d[o] = s_d[i]; n_e[o] = s_e[i]; n_l[o] = s_l[i]; }
            else       { n_id[o] = c_id[j]; n_d[o] = c_d[j]; n_e[o] = 0;      n_l[o] = c_l[j]; }
            if (n_id[o] < 0) { n_d[o] = FLT_MAX; n_e[o] = 1; n_l[o] = 0; }   // canonical empty
        }
        __syncthreads();
        for (int i = tid; i < L; i += BLOCK) {
            s_id[i] = n_id[i]; s_d[i] = n_d[i]; s_e[i] = n_e[i]; s_l[i] = n_l[i];
        }
        if (bound && tid == 0 && kstop > 0 && kstop <= L && n_id[kstop - 1] >= 0)
            bound_min(bound + q, n_d[kstop - 1]);
        __syncthreads();
    }

    for (int i = tid; i < L; i += BLOCK) {
        qid[i] = s_id[i]; qd[i] = s_d[i]; qe[i] = s_e[i]; ql[i] = s_l[i];
    }
}

#define DISPATCH_T(T, TEAMVAR, KERNEL, GRID, SHMEM, STREAM, XPTR, ...)                 \
    do {                                                                               \
        switch (TEAMVAR) {                                                             \
            case 4:  KERNEL<4,  T><<<GRID, BLOCK, SHMEM, STREAM>>>(XPTR, __VA_ARGS__); break; \
            case 8:  KERNEL<8,  T><<<GRID, BLOCK, SHMEM, STREAM>>>(XPTR, __VA_ARGS__); break; \
            case 16: KERNEL<16, T><<<GRID, BLOCK, SHMEM, STREAM>>>(XPTR, __VA_ARGS__); break; \
            case 32: KERNEL<32, T><<<GRID, BLOCK, SHMEM, STREAM>>>(XPTR, __VA_ARGS__); break; \
            default: TORCH_CHECK(false, "team_size must be 4, 8, 16 or 32");           \
        }                                                                              \
    } while (0)

// X may be fp32 or fp16; everything else is identical and the result differs
// only by the rounding of the stored vectors.
#define DISPATCH_TEAM(TEAMVAR, KERNEL, GRID, SHMEM, STREAM, X, ...)                    \
    do {                                                                               \
        if ((X).scalar_type() == torch::kFloat32)                                      \
            DISPATCH_T(float, TEAMVAR, KERNEL, GRID, SHMEM, STREAM,                    \
                       (X).data_ptr<float>(), __VA_ARGS__);                            \
        else                                                                           \
            DISPATCH_T(__half, TEAMVAR, KERNEL, GRID, SHMEM, STREAM,                   \
                       (__half*)(X).data_ptr<at::Half>(), __VA_ARGS__);                \
    } while (0)

static void check_common(const torch::Tensor& X, const torch::Tensor& Qv,
                         const torch::Tensor& ids, const torch::Tensor& dist,
                         const torch::Tensor& expd, const torch::Tensor& visited,
                         const torch::Tensor& counters) {
    TORCH_CHECK(X.is_cuda() && X.is_contiguous() &&
                (X.scalar_type() == torch::kFloat32 || X.scalar_type() == torch::kHalf),
                "X: contiguous cuda float32 or float16");
    TORCH_CHECK(Qv.is_cuda() && Qv.scalar_type() == torch::kFloat32 && Qv.is_contiguous(), "Q: contiguous cuda float32");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous(), "ids: contiguous int32");
    TORCH_CHECK(dist.scalar_type() == torch::kFloat32 && dist.is_contiguous(), "dist: contiguous float32");
    TORCH_CHECK(expd.scalar_type() == torch::kUInt8 && expd.is_contiguous(), "expanded: contiguous uint8");
    TORCH_CHECK(visited.scalar_type() == torch::kInt32 && visited.is_contiguous(), "visited: contiguous int32");
    TORCH_CHECK(counters.scalar_type() == torch::kInt32 && counters.is_contiguous(), "counters: contiguous int32");
}

void init_states(torch::Tensor X, torch::Tensor Qv, torch::Tensor seeds,
                 torch::Tensor ids, torch::Tensor dist, torch::Tensor expd,
                 torch::Tensor lin, torch::Tensor visited, torch::Tensor counters,
                 int64_t metric, int64_t team_size) {
    check_common(X, Qv, ids, dist, expd, visited, counters);
    TORCH_CHECK(lin.scalar_type() == torch::kUInt8 && lin.is_contiguous(), "lineage: contiguous uint8");
    const int dim = X.size(1), J = ids.size(1), L = ids.size(2), H = visited.size(2);
    const int V = seeds.dim() == 3 ? (int)seeds.size(2) : 1;
    TORCH_CHECK(V <= MAXD, "seeds per branch exceed MAXD (the init sort buffer)");
    TORCH_CHECK(V <= 256, "seeds per branch exceed 256 (lineage is a uint8)");
    TORCH_CHECK(L <= MAXL, "L exceeds MAXL");
    TORCH_CHECK((H & (H - 1)) == 0, "visited table size must be a power of two");
    TORCH_CHECK(counters.size(2) == NCTR, "counters last dim must be NCTR");
    dim3 grid(J, ids.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, init_kernel, grid, dim * sizeof(float), stream,
                  X, Qv.data_ptr<float>(), seeds.data_ptr<int>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  lin.data_ptr<unsigned char>(), visited.data_ptr<int>(), counters.data_ptr<int>(),
                  dim, J, V, L, H, (int)metric);
    C10_CUDA_CHECK(cudaGetLastError());
}

void advance(torch::Tensor X, torch::Tensor G, torch::Tensor Qv,
             torch::Tensor ids, torch::Tensor dist, torch::Tensor expd,
             torch::Tensor lin, torch::Tensor visited, torch::Tensor counters,
             int64_t metric, int64_t team_size, int64_t delta, int64_t max_expansions,
             int64_t policy, int64_t V_in, double stop_eps, int64_t kstop,
             torch::Tensor cache_id, torch::Tensor cache_d, torch::Tensor bound,
             int64_t exclusive) {
    check_common(X, Qv, ids, dist, expd, visited, counters);
    TORCH_CHECK(lin.scalar_type() == torch::kUInt8 && lin.is_contiguous(), "lineage: contiguous uint8");
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == torch::kInt32 && G.is_contiguous(), "G: contiguous cuda int32");
    const int dim = X.size(1), D = G.size(1);
    const int J = ids.size(1), L = ids.size(2), H = visited.size(2);
    TORCH_CHECK(D <= MAXD, "graph degree exceeds MAXD");
    TORCH_CHECK(V_in <= 256, "lineage count exceeds 256 (lineage is a uint8)");
    TORCH_CHECK(L <= MAXL, "L exceeds MAXL");
    TORCH_CHECK((H & (H - 1)) == 0, "visited table size must be a power of two");
    const int Hc = (cache_id.numel() > 0) ? (int)cache_id.size(1) : 0;
    if (Hc) {
        TORCH_CHECK(cache_id.scalar_type() == torch::kInt32 && cache_id.is_contiguous() &&
                    cache_d.scalar_type() == torch::kFloat32 && cache_d.is_contiguous(),
                    "cache: contiguous int32 / float32");
        TORCH_CHECK((Hc & (Hc - 1)) == 0, "cache size must be a power of two");
    }
    dim3 grid(J, ids.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, advance_kernel, grid, dim * sizeof(float), stream,
                  X, G.data_ptr<int>(), Qv.data_ptr<float>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  lin.data_ptr<unsigned char>(), visited.data_ptr<int>(), counters.data_ptr<int>(),
                  dim, D, J, L, H, (int)metric, (int)delta, (int)max_expansions,
                  (int)policy, (int)V_in, (float)stop_eps, (int)kstop,
                  Hc ? cache_id.data_ptr<int>() : nullptr,
                  Hc ? cache_d.data_ptr<float>() : nullptr, Hc,
                  bound.numel() ? bound.data_ptr<float>() : nullptr, (int)exclusive);
    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("init_states", &init_states, "initialise per-branch search states");
    mod.def("advance",     &advance,     "advance each branch by delta expansions");
}
