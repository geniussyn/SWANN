// Common local-search engine for multi-entry graph ANN (stage 1 of the design).
//
// One CTA per (query, branch). Each branch owns an independent best-m queue, an
// independent open-addressing visited table, and its own counters. The queue is
// kept sorted ascending by the total order (distance, id), and the `expanded`
// flag travels with its entry through every merge. A branch stops when its
// expansion budget is spent or when no retained candidate is left unexpanded --
// never merely because a round failed to change the queue.
//
// All state lives in caller-owned tensors, so `advance` is resumable: running
// delta=a then delta=b must equal running delta=a+b from the same init. That
// equality is what stage 2's probe/resume depends on, and it is a test here.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK   128
#define MAXM    64    // queue width ceiling (m <= MAXM)
#define MAXDEG  128   // graph out-degree ceiling (deg <= MAXDEG), == BLOCK for the sort
#define NCTR    4     // [0] expansions, [1] distance evals, [2] hash failures, [3] exhausted

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

// One distance, computed by a team of TEAM cooperating lanes. The reduction is a
// strided per-lane sum followed by a halving shuffle tree; refcpu.py replays
// exactly this order, which is what makes bitwise CPU/GPU agreement testable.
template <int TEAM>
__device__ __forceinline__ float team_dist(const float* __restrict__ q,
                                           const float* __restrict__ x,
                                           int D, int metric, int lane, unsigned tmask) {
    float acc = 0.f;
    if (metric == METRIC_IP) {
        for (int t = lane; t < D; t += TEAM) acc += q[t] * x[t];
    } else {
        for (int t = lane; t < D; t += TEAM) { float df = q[t] - x[t]; acc += df * df; }
    }
    for (int off = TEAM / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(tmask, acc, off, TEAM);
    return (metric == METRIC_IP) ? (1.f - acc) : acc;   // meaningful in lane 0
}

template <int TEAM>
__global__ void init_kernel(const float* __restrict__ X,
                            const float* __restrict__ Qv,
                            const int*   __restrict__ seeds,
                            int* ids, float* dist, unsigned char* expd,
                            int* visited, int* counters,
                            int D, int J, int m, int H, int metric) {
    const int j = blockIdx.x, q = blockIdx.y;
    const int tid = threadIdx.x, team = tid / TEAM, lane = tid % TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    extern __shared__ float sq[];                    // D floats: the query vector

    const size_t bid = (size_t)q * J + j;
    int*  vis = visited  + bid * H;
    int*  ctr = counters + bid * NCTR;
    int*  qid = ids      + bid * m;
    float* qd = dist     + bid * m;
    unsigned char* qe = expd + bid * m;

    for (int t = tid; t < D; t += BLOCK) sq[t] = Qv[(size_t)q * D + t];
    for (int t = tid; t < H; t += BLOCK) vis[t] = 0;
    for (int i = tid; i < m; i += BLOCK) { qid[i] = -1; qd[i] = FLT_MAX; qe[i] = 1; }
    for (int i = tid; i < NCTR; i += BLOCK) ctr[i] = 0;
    __syncthreads();

    const int s = seeds[bid];
    if (s >= 0) {
        if (team == 0) {
            float d = team_dist<TEAM>(sq, X + (size_t)s * D, D, metric, lane, tmask);
            if (lane == 0) {
                hash_insert(vis, H, s, &ctr[2]);
                qid[0] = s; qd[0] = d; qe[0] = 0;
                ctr[1] = 1;
            }
        }
    }
}

template <int TEAM>
__global__ void advance_kernel(const float* __restrict__ X,
                               const int*   __restrict__ G,
                               const float* __restrict__ Qv,
                               int* ids, float* dist, unsigned char* expd,
                               int* visited, int* counters,
                               int D, int deg, int J, int m, int H,
                               int metric, int delta, int max_expansions) {
    const int j = blockIdx.x, q = blockIdx.y;
    const int tid = threadIdx.x;
    const int team = tid / TEAM, lane = tid % TEAM, nteams = BLOCK / TEAM;
    const unsigned tmask = team_mask<TEAM>(tid);
    extern __shared__ float sq[];

    __shared__ int   s_id[MAXM];  __shared__ float s_d[MAXM];  __shared__ unsigned char s_e[MAXM];
    __shared__ int   n_id[MAXM];  __shared__ float n_d[MAXM];  __shared__ unsigned char n_e[MAXM];
    __shared__ int   c_id[MAXDEG]; __shared__ float c_d[MAXDEG];
    __shared__ int   s_slot, s_p;

    const size_t bid = (size_t)q * J + j;
    int*  vis = visited  + bid * H;
    int*  ctr = counters + bid * NCTR;
    int*  qid = ids      + bid * m;
    float* qd = dist     + bid * m;
    unsigned char* qe = expd + bid * m;

    for (int t = tid; t < D; t += BLOCK) sq[t] = Qv[(size_t)q * D + t];
    for (int i = tid; i < m; i += BLOCK) { s_id[i] = qid[i]; s_d[i] = qd[i]; s_e[i] = qe[i]; }
    __syncthreads();

    for (int step = 0; step < delta; ++step) {
        if (ctr[0] >= max_expansions) break;          // uniform: ctr[0] only moves in tid 0
        if (tid == 0) {
            s_slot = -1;
            for (int i = 0; i < m; ++i)
                if (s_id[i] >= 0 && s_e[i] == 0) { s_slot = i; break; }
            if (s_slot >= 0) { s_e[s_slot] = 1; s_p = s_id[s_slot]; ctr[0] += 1; }
            else             { ctr[3] = 1; }
        }
        __syncthreads();
        if (s_slot < 0) break;
        const int p = s_p;

        for (int i = tid; i < MAXDEG; i += BLOCK) { c_id[i] = -1; c_d[i] = FLT_MAX; }
        __syncthreads();

        // Expand p. Candidate i keeps the fixed slot i of G[p], so which lane did
        // the work never changes the result.
        for (int base = 0; base < deg; base += nteams) {
            const int nidx = base + team;
            if (nidx < deg) {
                const int v = G[(size_t)p * deg + nidx];
                if (v >= 0) {
                    int isnew = 0;
                    if (lane == 0) isnew = hash_insert(vis, H, v, &ctr[2]);
                    isnew = __shfl_sync(tmask, isnew, 0, TEAM);
                    if (isnew) {
                        float d = team_dist<TEAM>(sq, X + (size_t)v * D, D, metric, lane, tmask);
                        if (lane == 0) { c_id[nidx] = v; c_d[nidx] = d; atomicAdd(&ctr[1], 1); }
                    }
                }
            }
            __syncthreads();
        }

        // Bitonic sort of the MAXDEG candidate slots by (distance, id); the
        // never-filled slots carry FLT_MAX / -1 and land at the end.
        for (int k = 2; k <= MAXDEG; k <<= 1) {
            for (int s = k >> 1; s > 0; s >>= 1) {
                const int i = tid, ixj = tid ^ s;
                if (ixj > i) {
                    const bool asc = ((i & k) == 0);
                    const bool need = asc ? lt(c_d[ixj], c_id[ixj], c_d[i], c_id[i])
                                          : lt(c_d[i], c_id[i], c_d[ixj], c_id[ixj]);
                    if (need) {
                        float td = c_d[i]; c_d[i] = c_d[ixj]; c_d[ixj] = td;
                        int   ti = c_id[i]; c_id[i] = c_id[ixj]; c_id[ixj] = ti;
                    }
                }
                __syncthreads();
            }
        }

        // Two sorted runs -> best m. Serial and in one thread: it is 80 elements
        // against deg full-dimension distances, and it is the merge the CPU
        // reference performs verbatim.
        if (tid == 0) {
            int a = 0, b = 0, o = 0;
            while (o < m) {
                const bool ta = (a < m)      && (s_id[a] >= 0);
                const bool tb = (b < MAXDEG) && (c_id[b] >= 0);
                if (!ta && !tb) break;
                const bool takeA = (ta && tb) ? lt(s_d[a], s_id[a], c_d[b], c_id[b]) : ta;
                if (takeA) { n_id[o] = s_id[a]; n_d[o] = s_d[a]; n_e[o] = s_e[a]; ++a; }
                else       { n_id[o] = c_id[b]; n_d[o] = c_d[b]; n_e[o] = 0;      ++b; }
                ++o;
            }
            for (; o < m; ++o) { n_id[o] = -1; n_d[o] = FLT_MAX; n_e[o] = 1; }
        }
        __syncthreads();
        for (int i = tid; i < m; i += BLOCK) { s_id[i] = n_id[i]; s_d[i] = n_d[i]; s_e[i] = n_e[i]; }
        __syncthreads();
    }

    for (int i = tid; i < m; i += BLOCK) { qid[i] = s_id[i]; qd[i] = s_d[i]; qe[i] = s_e[i]; }
}

#define DISPATCH_TEAM(TEAMVAR, KERNEL, GRID, SHMEM, STREAM, ...)                       \
    do {                                                                       \
        switch (TEAMVAR) {                                                     \
            case 4:  KERNEL<4> <<<GRID, BLOCK, SHMEM, STREAM>>>(__VA_ARGS__); break;   \
            case 8:  KERNEL<8> <<<GRID, BLOCK, SHMEM, STREAM>>>(__VA_ARGS__); break;   \
            case 16: KERNEL<16><<<GRID, BLOCK, SHMEM, STREAM>>>(__VA_ARGS__); break;   \
            case 32: KERNEL<32><<<GRID, BLOCK, SHMEM, STREAM>>>(__VA_ARGS__); break;   \
            default: TORCH_CHECK(false, "team_size must be 4, 8, 16 or 32");   \
        }                                                                      \
    } while (0)

static void check_common(const torch::Tensor& X, const torch::Tensor& Qv,
                         const torch::Tensor& ids, const torch::Tensor& dist,
                         const torch::Tensor& expd, const torch::Tensor& visited,
                         const torch::Tensor& counters) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.is_contiguous(), "X: contiguous cuda float32");
    TORCH_CHECK(Qv.is_cuda() && Qv.scalar_type() == torch::kFloat32 && Qv.is_contiguous(), "Q: contiguous cuda float32");
    TORCH_CHECK(ids.scalar_type() == torch::kInt32 && ids.is_contiguous(), "ids: contiguous int32");
    TORCH_CHECK(dist.scalar_type() == torch::kFloat32 && dist.is_contiguous(), "dist: contiguous float32");
    TORCH_CHECK(expd.scalar_type() == torch::kUInt8 && expd.is_contiguous(), "expanded: contiguous uint8");
    TORCH_CHECK(visited.scalar_type() == torch::kInt32 && visited.is_contiguous(), "visited: contiguous int32");
    TORCH_CHECK(counters.scalar_type() == torch::kInt32 && counters.is_contiguous(), "counters: contiguous int32");
}

void init_states(torch::Tensor X, torch::Tensor Qv, torch::Tensor seeds,
                 torch::Tensor ids, torch::Tensor dist, torch::Tensor expd,
                 torch::Tensor visited, torch::Tensor counters,
                 int64_t metric, int64_t team_size) {
    check_common(X, Qv, ids, dist, expd, visited, counters);
    const int D = X.size(1), Q = ids.size(0), J = ids.size(1), m = ids.size(2), H = visited.size(2);
    TORCH_CHECK(m <= MAXM, "m exceeds MAXM");
    TORCH_CHECK((H & (H - 1)) == 0, "visited table size must be a power of two");
    TORCH_CHECK(counters.size(2) == NCTR, "counters last dim must be 4");
    dim3 grid(J, Q);
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, init_kernel, grid, D * sizeof(float), stream,
                  X.data_ptr<float>(), Qv.data_ptr<float>(), seeds.data_ptr<int>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  visited.data_ptr<int>(), counters.data_ptr<int>(),
                  D, J, m, H, (int)metric);
    C10_CUDA_CHECK(cudaGetLastError());
}

void advance(torch::Tensor X, torch::Tensor G, torch::Tensor Qv,
             torch::Tensor ids, torch::Tensor dist, torch::Tensor expd,
             torch::Tensor visited, torch::Tensor counters,
             int64_t metric, int64_t team_size, int64_t delta, int64_t max_expansions) {
    check_common(X, Qv, ids, dist, expd, visited, counters);
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == torch::kInt32 && G.is_contiguous(), "G: contiguous cuda int32");
    const int D = X.size(1), deg = G.size(1);
    const int Q = ids.size(0), J = ids.size(1), m = ids.size(2), H = visited.size(2);
    TORCH_CHECK(deg <= MAXDEG, "graph degree exceeds MAXDEG");
    TORCH_CHECK(m <= MAXM, "m exceeds MAXM");
    TORCH_CHECK((H & (H - 1)) == 0, "visited table size must be a power of two");
    dim3 grid(J, Q);
    auto stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_TEAM(team_size, advance_kernel, grid, D * sizeof(float), stream,
                  X.data_ptr<float>(), G.data_ptr<int>(), Qv.data_ptr<float>(),
                  ids.data_ptr<int>(), dist.data_ptr<float>(), expd.data_ptr<unsigned char>(),
                  visited.data_ptr<int>(), counters.data_ptr<int>(),
                  D, deg, J, m, H, (int)metric, (int)delta, (int)max_expansions);
    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("init_states", &init_states, "initialise per-branch search states");
    mod.def("advance",     &advance,     "advance each branch by delta expansions");
}
