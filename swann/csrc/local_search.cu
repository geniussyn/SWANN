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
#define MAXL    512   // L, the queue width, may not exceed this
#define MAXD    128   // D, the graph out-degree, may not exceed this (== BLOCK)
#define NCTR   10     // [0] expansions, [1] distance evals, [2] hash failures,
                      // [3] exhausted, [4] cache hits, [5] cache slots claimed,
                      // [6] stopped because the k-th best stopped moving,
                      // [7] lineages retired for expanding into covered ground,
                      // [8] candidates that beat the queue's worst (effective
                      //     fan-out: how much of an adjacency list can matter),
                      // [9] expansions those were counted over
#define CPROBE  32    // bounded probe into the shared cache; a miss just recomputes

// H == 1 means the per-branch visited table is switched off. In exclusive mode
// the per-query cache already refuses a node that anyone -- including this
// branch -- has claimed, so the second table is redundant there, and dropping it
// removes one random global probe per candidate and the H ints per branch that
// have to be cleared before every search (22 GB a batch at V=384). What it gives
// up is the guarantee: a cache probe that runs out of room returns -1 and the
// node is computed and queued again, so a branch's queue can hold a duplicate.
// Measured equivalence is the thing to check, not assumed.
//
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

// An expansion reads one full vector per neighbour and keeps 12.5 of 96: 87% of
// the bytes it moves are spent deciding against a candidate. The decision needs
// far less than the vector. A sign code -- one bit per dimension, taken around
// the corpus mean -- answers it at 1/32 of the bytes, and with the node's own
// ||x - mu||_1 / d as a scale the estimate lands in the corpus's own units, so
// it can be compared against the queue's worst entry directly instead of having
// to be ranked against the other candidates.
//
// Measured on MS MARCO at the shipped operating point: skipping every candidate
// whose estimate exceeds the queue's worst by more than 0.20 lets 27.4% of the
// slots through and keeps 100% of the candidates the exact test would admit.
// The margin is in distance units and has to be set per corpus; at 0.30 it lets
// 85.6% through and buys nothing.
//
// <q, x> = alpha * <q, b> + <q, mu>, and with b in {-1, +1},
// <q, b> = 2 * (sum of q over set bits) - (sum of all q), so one pass over the
// code words suffices given the query's own sum.
template <int TEAM>
__device__ __forceinline__ float team_code_ip(const float* __restrict__ q,
                                              const int* __restrict__ code,
                                              float alpha, float qmu, float qsum,
                                              int dim, int cwords,
                                              int lane, unsigned tmask) {
    float acc = 0.f;
    for (int w = lane; w < cwords; w += TEAM) {
        const unsigned bits = (unsigned)code[w];
        const int base = w * 32;
        #pragma unroll
        for (int b = 0; b < 32; ++b) {
            const int i = base + b;
            if (i >= dim) break;
            if (bits & (1u << b)) acc += q[i];
        }
    }
    for (int off = TEAM / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(tmask, acc, off, TEAM);
    // meaningful in lane 0
    return 1.f - (alpha * (2.f * acc - qsum) + qmu);
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
                            int dim, int J, int V, int L, int H, int metric,
                            int* cache_id, float* cache_d, int Hc, float* bound) {
    const int j = blockIdx.x, q = blockIdx.y;
    const int tid = threadIdx.x, team = tid / TEAM, lane = tid % TEAM, nteams = BLOCK / TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    // One dynamic block, carved to the L and D actually in use. Sizing these
    // to MAXL/MAXD instead cost 6.3 KB a block where 0.9 KB was wanted, and
    // shared memory per SM is what caps how many blocks run at once.
    extern __shared__ char smem[];
    int nsort = 2; while (nsort < V && nsort < MAXD) nsort <<= 1;
    int*   c_id = (int*)smem;
    float* c_d  = (float*)(c_id + nsort);
    float* sq   = c_d + nsort;                       // dim floats: the query vector
    unsigned char* c_l = (unsigned char*)(sq + dim);

    const size_t bid = (size_t)q * J + j;
    int*  vis = visited  + bid * H;
    int*  ctr = counters + bid * NCTR;
    int*  qid = ids      + bid * L;
    float* qd = dist     + bid * L;
    unsigned char* qe = expd + bid * L;
    unsigned char* ql = lin  + bid * L;

    for (int t = tid; t < dim; t += BLOCK) sq[t] = Qv[(size_t)q * dim + t];
    for (int t = tid; t < H; t += BLOCK) vis[t] = 0;    // H == 1: one word, free
    // The per-query cache and bound belong to the query, not the branch, so the
    // J blocks of one query clear it between them. Doing it here rather than
    // from the host removes two full-tensor fills per search -- 3.6 GB at
    // Q = 3490, V = 128, which was most of the 7.8% spent initialising.
    if (Hc > 0) {
        int* cid0 = cache_id + (size_t)q * Hc;
        float* cdd0 = cache_d + (size_t)q * Hc;
        for (int t = j * BLOCK + tid; t < Hc; t += J * BLOCK) {
            cid0[t] = 0; cdd0[t] = FLT_MAX;
        }
    }
    if (bound && j == 0 && tid == 0) bound[q] = FLT_MAX;
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
                int isnew = 1;
                if (H > 1) {
                    if (lane == 0) isnew = hash_insert(vis, H, v, &ctr[2]);
                    isnew = __shfl_sync(tmask, isnew, 0, TEAM);
                }
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
                               int exclusive, int stall_patience,
                               float crowd_frac, const int* active,
                               const int* __restrict__ codes,
                               const float* __restrict__ alpha,
                               const float* __restrict__ cmu,
                               int cwords, float cmargin) {
    // With `active`, the grid is a compacted list of the (query, branch) pairs
    // still running rather than the full J x Q rectangle. Branches finish at
    // very different times -- mean 5 expansions against a maximum of 25 -- so a
    // rectangular grid keeps launching blocks that exit on their first test
    // while the slowest block in flight holds its slot. Re-forming the grid
    // between chunks is what chunked-prefill does for a batch of uneven
    // requests, and the state here is already resumable, so it costs nothing
    // but the launch.
    const int flat = active ? active[blockIdx.x] : (int)(blockIdx.y * gridDim.x + blockIdx.x);
    const int j = active ? (flat % J) : (int)blockIdx.x;
    const int q = active ? (flat / J) : (int)blockIdx.y;
    const int tid = threadIdx.x;
    const int team = tid / TEAM, lane = tid % TEAM, nteams = BLOCK / TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    // One dynamic block, carved to the L and D actually in use (see init_kernel).
    extern __shared__ char smem[];
    int nsort = 2; while (nsort < D && nsort < MAXD) nsort <<= 1;
    int*   s_id = (int*)smem;
    int*   n_id = s_id + L;
    int*   c_id = n_id + L;
    float* s_d  = (float*)(c_id + nsort);
    float* n_d  = s_d + L;
    float* c_d  = n_d + L;
    float* sq   = c_d + nsort;
    unsigned char* s_e = (unsigned char*)(sq + dim);
    unsigned char* s_l = s_e + L;
    unsigned char* n_e = s_l + L;
    unsigned char* n_l = n_e + L;
    unsigned char* c_l = n_l + L;
    __shared__ int   s_slot, s_p, s_plin;
    // 256 lineage bits, because lineage is a uint8 and V may be up to 256. This
    // was one 32-bit word, which silently shifted out of range for V > 32 -- the
    // measurements at V = 32 sat exactly on that boundary.
    __shared__ unsigned s_retired[8];
    // Stalling is a different signal from stop_eps: eps asks "is the frontier
    // too far", this asks "has the answer stopped moving". A hard query keeps a
    // close frontier for a long time without improving, and only this catches it.
    __shared__ float s_best_k;
    __shared__ int   s_stall;
    // How much of an expansion landed on ground already covered -- this branch's
    // own visited set, or a node another branch has claimed. The cache knows this
    // for free, and it says "you are retreading" directly, where stop_eps can only
    // ask "are you far". A branch whose neighbours are all already taken has
    // nowhere to go, and spending its remaining budget is what makes the
    // uncoordinated arm saturate.
    __shared__ int s_nbr, s_cov, s_crowd_run;
    __shared__ int s_nkeep;        // candidates that can still reach the queue

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
    // <q, mu> and sum(q) are the same for every candidate of this query, so the
    // estimator needs them once rather than per neighbour.
    __shared__ float s_qmu, s_qsum, s_rad;
    if (tid == 0) { s_qmu = 0.f; s_qsum = 0.f; }
    __syncthreads();
    if (codes) {
        float a = 0.f, b = 0.f;
        for (int t = tid; t < dim; t += BLOCK) {
            const float qt = Qv[(size_t)q * dim + t];
            a += qt * cmu[t]; b += qt;
        }
        atomicAdd(&s_qmu, a); atomicAdd(&s_qsum, b);
    }
    __syncthreads();                       // s_qmu / s_qsum are read below
    for (int i = tid; i < L; i += BLOCK) {
        s_id[i] = qid[i]; s_d[i] = qd[i]; s_e[i] = qe[i]; s_l[i] = ql[i];
    }
    // Retirement is re-derived every call rather than stored: it is a function
    // of the state, it is monotone, and a re-check costs no expansion.
    for (int i = tid; i < 8; i += BLOCK) s_retired[i] = 0u;
    if (tid == 0) { s_best_k = FLT_MAX; s_stall = 0; s_crowd_run = 0; }
    __syncthreads();

    for (int step = 0; step < delta; ++step) {
        if (ctr[0] >= max_expansions) break;          // uniform: ctr[0] only moves in tid 0
        if (tid == 0) {
            s_slot = -1;
            const bool early = (stop_eps >= 0.f);
            // The radius is the k-th best this branch holds -- or its L-th, when
            // the queue is narrower than k. L < k is not degenerate: the answer
            // is merged across V branches, so a branch holding fewer than k
            // results is contributing its share and nothing is lost. What does
            // change is the rule's strictness: the L-th best is nearer than the
            // k-th would be, so the radius is tighter and lineages retire
            // sooner. That is a shift along the recall/work curve, which eps is
            // there to move, not a change in what the search may return.
            const int kr = (kstop < L) ? kstop : L;
            float radius = (kr > 0 && s_id[kr - 1] >= 0) ? s_d[kr - 1] : FLT_MAX;
            if (bound) {                      // what the whole query knows, not this branch
                const float gb = ((volatile float*)bound)[q];
                if (gb < radius) radius = gb;
            }
            const float limit = (radius == FLT_MAX) ? FLT_MAX : radius * (1.f + stop_eps);
            // The code filter needs this one, not the radius. The queue is a
            // frontier and not just a result list: a candidate worse than the
            // k-th best can still be the node that leads to a better one, and
            // dropping those costs recall (-0.0010 measured when the radius was
            // used). `limit` is the line the search itself will not expand past,
            // so a candidate beyond it plus a margin is one the search was
            // never going to follow.
            s_rad = limit;

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
                        // s_retired is all zeros unless a crowd or round-robin
                        // retirement set it, so this is a no-op by default.
                        const unsigned li = s_l[i];
                        if (s_retired[li >> 5] & (1u << (li & 31))) continue;
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
        if (tid == 0) { s_nbr = 0; s_cov = 0; }
        __syncthreads();

        // Expand p. Candidate i keeps the fixed slot i of G[p], so which lane did
        // the work never changes the result, and it inherits p's lineage.
        for (int base = 0; base < D; base += nteams) {
            const int nidx = base + team;
            if (nidx < D) {
                const int v = G[(size_t)p * D + nidx];
                if (v >= 0) {
                    if (lane == 0) atomicAdd(&s_nbr, 1);
                    int isnew = 1;
                    if (H > 1) {
                        if (lane == 0) isnew = hash_insert(vis, H, v, &ctr[2]);
                        isnew = __shfl_sync(tmask, isnew, 0, TEAM);
                    }
                    if (!isnew) { if (lane == 0) atomicAdd(&s_cov, 1); }
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
                            // Decide from the code before paying for the vector:
                            // a candidate the estimate puts beyond the queue's
                            // worst by more than the margin cannot enter it, so
                            // its vector is never read. The result is a skip
                            // rather than a `continue` -- the neighbour loop
                            // closes on a __syncthreads() that every team has to
                            // reach, and teams diverge here.
                            bool skip = false;
                            if (codes) {
                                float est = team_code_ip<TEAM>(
                                    sq, codes + (size_t)v * cwords, alpha[v],
                                    s_qmu, s_qsum, dim, cwords, lane, tmask);
                                est = __shfl_sync(tmask, est, 0, TEAM);
                                skip = (s_rad != FLT_MAX) && (est > s_rad + cmargin);
                                if (skip && lane == 0) atomicAdd(&ctr[3], 1);
                            }
                            if (!skip) {
                                d = team_dist<TEAM, XT>(sq, X + (size_t)v * dim,
                                                        dim, metric, lane, tmask);
                                if (lane == 0) {
                                    if (slot >= 0) {
                                        ((volatile float*)cdd)[slot] = d;
                                        atomicAdd(&ctr[5], 1);
                                    }
                                    atomicAdd(&ctr[1], 1);
                                }
                            }
                        } else if (lane == 0) {
                            atomicAdd(&ctr[4], 1);
                        }
                        const bool taken = exclusive && slot >= 0 && !claimed && d != FLT_MAX;
                        if (lane == 0 && taken) atomicAdd(&s_cov, 1);
                        if (lane == 0 && !taken) {
                            c_id[nidx] = v; c_d[nidx] = d; c_l[nidx] = plin;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Sort only what can still reach the queue. The builder paid for D
        // neighbours, but a candidate no better than the queue's worst cannot
        // enter it, and measurement puts that live fraction at 10-16 of 64. A
        // bitonic sort is (log n)(log n + 1)/2 barriers, so cutting n from 64 to
        // 16 cuts 21 stages to 10. nsort <= MAXD == BLOCK, so each thread owns
        // exactly one slot and the compaction needs no second buffer.
        //
        // <= rather than <: lt() breaks a distance tie by id, so a candidate at
        // exactly the worst distance can still displace it.
        {
            const float worst = (s_id[L - 1] >= 0) ? s_d[L - 1] : FLT_MAX;
            int kv = -1; float kd = FLT_MAX; unsigned char kl = 0;
            bool keep = false;
            if (tid < nsort) {
                kv = c_id[tid]; kd = c_d[tid]; kl = c_l[tid];
                keep = (kv >= 0) && (worst == FLT_MAX || kd <= worst);
            }
            if (tid == 0) s_nkeep = 0;
            __syncthreads();
            int pos = -1;
            if (keep) pos = atomicAdd(&s_nkeep, 1);
            __syncthreads();
            if (tid < nsort) { c_id[tid] = -1; c_d[tid] = FLT_MAX; c_l[tid] = 0; }
            __syncthreads();
            if (keep) { c_id[pos] = kv; c_d[pos] = kd; c_l[pos] = kl; }
            if (tid == 0) { ctr[8] += s_nkeep; ctr[9] += 1; }
            __syncthreads();
        }
        int msort = 2; while (msort < s_nkeep) msort <<= 1;
        if (msort > nsort) msort = nsort;
        bitonic_sort(c_id, c_d, c_l, tid, msort);

        // Two sorted runs -> best L, one output slot per thread.
        //
        // The serial two-pointer merge this replaces ran up to L + D steps in
        // thread 0 with the other 127 idle, every expansion. Merge-path instead:
        // output slot o is fed by exactly i elements of the queue and o - i of
        // the candidates, and that i is found by a binary search of log(L) steps
        // that each thread can run on its own. Same output, same tie-break.
        for (int o = tid; o < L; o += BLOCK) {
            int lo = max(0, o - msort), hi = min(o, L);
            while (lo < hi) {
                const int i = (lo + hi) >> 1, j = o - i;
                if (j > 0 && !lt(c_d[j - 1], c_id[j - 1], s_d[i], s_id[i])) lo = i + 1;
                else                                                       hi = i;
            }
            const int i = lo, j = o - i;
            const bool takeA = (j >= msort) ||
                               (i < L && lt(s_d[i], s_id[i], c_d[j], c_id[j]));
            if (takeA) { n_id[o] = s_id[i]; n_d[o] = s_d[i]; n_e[o] = s_e[i]; n_l[o] = s_l[i]; }
            else       { n_id[o] = c_id[j]; n_d[o] = c_d[j]; n_e[o] = 0;      n_l[o] = c_l[j]; }
            if (n_id[o] < 0) { n_d[o] = FLT_MAX; n_e[o] = 1; n_l[o] = 0; }   // canonical empty
        }
        __syncthreads();
        for (int i = tid; i < L; i += BLOCK) {
            s_id[i] = n_id[i]; s_d[i] = n_d[i]; s_e[i] = n_e[i]; s_l[i] = n_l[i];
        }
        if (bound && tid == 0) {
            const int kr = (kstop < L) ? kstop : L;      // see the radius above
            if (kr > 0 && n_id[kr - 1] >= 0) bound_min(bound + q, n_d[kr - 1]);
        }
        if (crowd_frac > 0.f && tid == 0) {
            const float cov = (s_nbr > 0) ? ((float)s_cov / (float)s_nbr) : 0.f;
            if (cov >= crowd_frac) {
                if (++s_crowd_run >= 2) {        // two in a row, not one unlucky step
                    const unsigned li = (unsigned)s_plin;
                    s_retired[li >> 5] |= (1u << (li & 31));
                    ctr[7] += 1;
                    s_crowd_run = 0;
                }
            } else {
                s_crowd_run = 0;
            }
        }
        if (stall_patience > 0 && tid == 0) {
            const int kr = (kstop < L) ? kstop : L;      // see the radius above
            const float bk = (kr > 0 && n_id[kr - 1] >= 0) ? n_d[kr - 1] : FLT_MAX;
            if (bk < s_best_k) { s_best_k = bk; s_stall = 0; } else { s_stall += 1; }
        }
        __syncthreads();
        // s_stall is shared and read after the barrier, so the break is uniform.
        if (stall_patience > 0 && s_stall >= stall_patience) {
            if (tid == 0) ctr[6] = 1;
            break;
        }
    }

    for (int i = tid; i < L; i += BLOCK) {
        qid[i] = s_id[i]; qd[i] = s_d[i]; qe[i] = s_e[i]; ql[i] = s_l[i];
    }
}

// Shared memory a block needs: the query vector, plus the queue and candidate
// buffers at their actual widths. Must match the carve-up in the kernels.
static inline size_t shmem_init(int dim, int V) {
    int nsort = 2; while (nsort < V && nsort < MAXD) nsort <<= 1;
    return (size_t)dim * sizeof(float) + (size_t)nsort * (4 + 4 + 1);
}
static inline size_t shmem_advance(int dim, int D, int L) {
    int nsort = 2; while (nsort < D && nsort < MAXD) nsort <<= 1;
    return (size_t)dim * sizeof(float)
         + (size_t)L * (4 + 4 + 1 + 1 + 4 + 4 + 1 + 1)   // s_* and n_*
         + (size_t)nsort * (4 + 4 + 1);                  // c_*
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
                 int64_t metric, int64_t team_size,
                 torch::Tensor cache_id, torch::Tensor cache_d, torch::Tensor bound) {
    check_common(X, Qv, ids, dist, expd, visited, counters);
    TORCH_CHECK(lin.scalar_type() == torch::kUInt8 && lin.is_contiguous(), "lineage: contiguous uint8");
    const int dim = X.size(1), J = ids.size(1), L = ids.size(2), H = visited.size(2);
    const int V = seeds.dim() == 3 ? (int)seeds.size(2) : 1;
    TORCH_CHECK(V <= MAXD, "seeds per branch exceed MAXD (the init sort buffer)");
    TORCH_CHECK(V <= 256, "seeds per branch exceed 256 (lineage is a uint8)");
    TORCH_CHECK(L <= MAXL, "L exceeds MAXL");
    TORCH_CHECK((H & (H - 1)) == 0, "visited table size must be a power of two");
    TORCH_CHECK(counters.size(2) == NCTR, "counters last dim must be NCTR");
    const int Hc = (cache_id.numel() > 0) ? (int)cache_id.size(1) : 0;
    const size_t shm = shmem_init(dim, V);
    TORCH_CHECK(shm <= 48u * 1024u, "init needs ", shm, " B of shared memory; the "
                "per-block default is 48 KB. Lower L or the dimension.");
    // The query index is grid.y, and CUDA caps y at 65535. Without this the
    // launch fails with "invalid configuration argument", which says nothing
    // about which limit was hit.
    TORCH_CHECK(ids.size(0) <= 65535,
                "batch of ", ids.size(0), " queries exceeds the 65535 the grid's "
                "y dimension allows; split the batch");
    dim3 grid(J, ids.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, init_kernel, grid, shm, stream,
                  X, Qv.data_ptr<float>(), seeds.data_ptr<int>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  lin.data_ptr<unsigned char>(), visited.data_ptr<int>(), counters.data_ptr<int>(),
                  dim, J, V, L, H, (int)metric,
                  Hc ? cache_id.data_ptr<int>() : nullptr,
                  Hc ? cache_d.data_ptr<float>() : nullptr, Hc,
                  bound.numel() ? bound.data_ptr<float>() : nullptr);
    C10_CUDA_CHECK(cudaGetLastError());
}

void advance(torch::Tensor X, torch::Tensor G, torch::Tensor Qv,
             torch::Tensor ids, torch::Tensor dist, torch::Tensor expd,
             torch::Tensor lin, torch::Tensor visited, torch::Tensor counters,
             int64_t metric, int64_t team_size, int64_t delta, int64_t max_expansions,
             int64_t policy, int64_t V_in, double stop_eps, int64_t kstop,
             torch::Tensor cache_id, torch::Tensor cache_d, torch::Tensor bound,
             int64_t exclusive, int64_t stall_patience, double crowd_frac,
             torch::Tensor active, torch::Tensor codes, torch::Tensor alpha,
             torch::Tensor cmu, double code_margin) {
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
    const size_t shm = shmem_advance(dim, D, L);
    TORCH_CHECK(shm <= 48u * 1024u, "advance needs ", shm, " B of shared memory; the "
                "per-block default is 48 KB. Lower L or the dimension.");
    const int nact = (int)active.numel();
    if (nact) {
        TORCH_CHECK(active.scalar_type() == torch::kInt32 && active.is_contiguous(),
                    "active: contiguous int32");
    }
    int ncode = 0;
    if (codes.numel()) {
        TORCH_CHECK(metric == METRIC_IP, "the sign-code filter is written for the "
                    "normalised inner-product corpora; an unnormalised L2 corpus "
                    "needs int8 codes and a stored squared norm");
        TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kInt32 &&
                    codes.is_contiguous(), "codes: contiguous cuda int32");
        TORCH_CHECK(alpha.numel() == codes.size(0) && cmu.numel() == dim,
                    "codes, alpha and mu must agree with the corpus");
        ncode = (int)codes.size(1);
        TORCH_CHECK(ncode * 32 >= dim, "code words do not cover the dimension");
    }
    dim3 grid = nact ? dim3(nact, 1) : dim3(J, ids.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, advance_kernel, grid, shm, stream,
                  X, G.data_ptr<int>(), Qv.data_ptr<float>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  lin.data_ptr<unsigned char>(), visited.data_ptr<int>(), counters.data_ptr<int>(),
                  dim, D, J, L, H, (int)metric, (int)delta, (int)max_expansions,
                  (int)policy, (int)V_in, (float)stop_eps, (int)kstop,
                  Hc ? cache_id.data_ptr<int>() : nullptr,
                  Hc ? cache_d.data_ptr<float>() : nullptr, Hc,
                  bound.numel() ? bound.data_ptr<float>() : nullptr, (int)exclusive,
                  (int)stall_patience, (float)crowd_frac,
                  nact ? active.data_ptr<int>() : nullptr,
                  ncode ? codes.data_ptr<int>() : nullptr,
                  ncode ? alpha.data_ptr<float>() : nullptr,
                  ncode ? cmu.data_ptr<float>() : nullptr,
                  ncode, (float)code_margin);
    C10_CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// Merging the branches of one query into its answer.
//
// The Python path did this in six tensor operations -- argsort by id, gather,
// a shifted comparison for duplicates, masked_fill, topk, gather -- which at
// batch one cost a fifth of the whole search, because each is a launch over a
// few hundred elements. One block per query does it in one.
//
// Duplicates arise because branches share a distance table: a node two branches
// both hold was scored once, so its copies carry the same distance, and
// dropping all but one of them is a dedup and not a choice.

#define MERGE_BLOCK 256

__device__ __forceinline__ bool merge_less(float da, int ia, float db, int ib) {
    // (distance, id) order, so copies of one id land adjacent
    return (da < db) || (da == db && ia < ib);
}

__global__ void merge_branches_kernel(
        const int* __restrict__ bids,     // (Q, V, kk)
        const float* __restrict__ bdist,
        int* __restrict__ out_ids,        // (Q, k)
        float* __restrict__ out_dist,
        int V, int kk, int k, int padded) {
    extern __shared__ char smem_merge[];
    float* sd = reinterpret_cast<float*>(smem_merge);
    int*   si = reinterpret_cast<int*>(sd + padded);

    const int q = blockIdx.x;
    const int n = V * kk;
    const int base = q * n;

    for (int i = threadIdx.x; i < padded; i += MERGE_BLOCK) {
        bool real = i < n;
        int id = real ? bids[base + i] : -1;
        float d = (real && id >= 0) ? bdist[base + i] : INFINITY;
        si[i] = (id >= 0) ? id : INT_MAX;   // INT_MAX so invalids sort last
        sd[i] = d;
    }
    __syncthreads();

    // bitonic sort over the padded power of two
    for (int size = 2; size <= padded; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < padded; i += MERGE_BLOCK) {
                int j = i ^ stride;
                if (j > i) {
                    bool up = ((i & size) == 0);
                    bool lt = merge_less(sd[i], si[i], sd[j], si[j]);
                    if (up != lt) {
                        float td = sd[i]; sd[i] = sd[j]; sd[j] = td;
                        int   ti = si[i]; si[i] = si[j]; si[j] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    // a run of equal ids is now contiguous; keep its first element only
    if (threadIdx.x == 0) {
        int w = 0;
        for (int i = 0; i < n && w < k; ++i) {
            if (si[i] == INT_MAX) break;               // nothing valid left
            if (i > 0 && si[i] == si[i - 1]) continue; // a copy of the previous
            out_ids[q * k + w] = si[i];
            out_dist[q * k + w] = sd[i];
            ++w;
        }
        for (; w < k; ++w) {
            out_ids[q * k + w] = -1;
            out_dist[q * k + w] = INFINITY;
        }
    }
}

static inline int next_pow2_int(int x) {
    int p = 1; while (p < x) p <<= 1; return p;
}

std::vector<at::Tensor> merge_branches(at::Tensor bids, at::Tensor bdist, int64_t k) {
    TORCH_CHECK(bids.is_cuda() && bdist.is_cuda(), "merge inputs must be on the GPU");
    TORCH_CHECK(bids.dim() == 3 && bdist.sizes() == bids.sizes(),
                "expected (Q, V, kk) for both ids and distances");
    TORCH_CHECK(bids.scalar_type() == at::kInt, "ids must be int32");
    TORCH_CHECK(bdist.scalar_type() == at::kFloat, "distances must be float32");
    bids = bids.contiguous(); bdist = bdist.contiguous();
    const int Q = (int)bids.size(0), V = (int)bids.size(1), kk = (int)bids.size(2);
    const int n = V * kk, padded = next_pow2_int(n);
    TORCH_CHECK(k > 0 && k <= n, "k must be in 1..V*kk");
    const size_t shm = (size_t)padded * (sizeof(float) + sizeof(int));
    TORCH_CHECK(shm <= 48u * 1024u, "merge needs ", shm, " B of shared memory for ",
                n, " candidates; lower V or the per-branch k");

    auto oi = at::empty({Q, (int64_t)k}, bids.options());
    auto od = at::empty({Q, (int64_t)k}, bdist.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    merge_branches_kernel<<<Q, MERGE_BLOCK, shm, stream>>>(
        bids.data_ptr<int>(), bdist.data_ptr<float>(),
        oi.data_ptr<int>(), od.data_ptr<float>(), V, kk, (int)k, padded);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {oi, od};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("init_states", &init_states, "initialise per-branch search states");
    mod.def("advance",     &advance,     "advance each branch by delta expansions");
    mod.def("merge_branches", &merge_branches,
            "dedup by id and take the k nearest, one block per query");
}
