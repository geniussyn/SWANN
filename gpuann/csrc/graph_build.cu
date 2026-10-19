// Query-free fragmentation, measured per node, so a builder can spend degree
// where the graph is about to fail.
//
// varphi_h(q) -- the manuscript's fragmentation index -- counts the components
// of a query's exact top-k under mutual h-hop reachability. It needs a query.
// A builder has none, so it asks the same question of the corpus instead: take
// a document's own top-k neighbourhood as a stand-in answer set and count how
// many pieces the graph cuts *it* into. A node whose own neighbourhood is
// already in several pieces sits where the graph is thin, and that is where an
// extra edge buys something.
//
// The kernel computes h=2 mutual reachability for every node at once. The naive
// form -- one BFS per member -- re-reads the same adjacency millions of times.
// Instead one combined walk carries a k-bit mask of "which members reached
// here": level 1 tags each neighbour with its member's bit, level 2 expands only
// those tagged nodes and ORs their masks onto any member it lands on. Work per
// node falls from k*deg^2 to roughly k*deg edge reads.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define BB        128     // threads per block
#define MAXK      32      // neighbourhood size ceiling (mask is one uint32)
#define L1CAP     2048    // open-addressed slots for the level-1 node set

__device__ __forceinline__ int l1_slot(int* keys, unsigned* masks, int key,
                                       unsigned bit, int* overflow) {
    unsigned pos = (2654435761u * (unsigned)key) & (L1CAP - 1);
    for (int probe = 0; probe < L1CAP; ++probe) {
        int cur = keys[pos];
        if (cur == key + 1) { atomicOr(&masks[pos], bit); return (int)pos; }
        if (cur == 0) {
            int old = atomicCAS(&keys[pos], 0, key + 1);
            if (old == 0 || old == key + 1) { atomicOr(&masks[pos], bit); return (int)pos; }
        }
        pos = (pos + 1) & (L1CAP - 1);
    }
    atomicAdd(overflow, 1);
    return -1;
}

// members[N, K]: the neighbourhood of each node (its own top-K, u itself excluded).
// reach[N, K]:   bit i of reach[u][j] == member i reaches member j within h hops.
__global__ void neighborhood_reach_kernel(const int* __restrict__ G,
                                          const int* __restrict__ members,
                                          unsigned* reach, int* overflow,
                                          int N, int deg, int K, int h) {
    const int u = blockIdx.x;
    if (u >= N) return;
    const int tid = threadIdx.x;

    __shared__ int   mem[MAXK];
    __shared__ unsigned rch[MAXK];
    __shared__ int   l1key[L1CAP];
    __shared__ unsigned l1mask[L1CAP];

    for (int i = tid; i < K; i += BB) { mem[i] = members[(size_t)u * K + i]; rch[i] = 0u; }
    for (int i = tid; i < L1CAP; i += BB) { l1key[i] = 0; l1mask[i] = 0u; }
    __syncthreads();

    // Level 1: every member's out-edges. A member landing on a member is a
    // one-hop reach; everything else is tagged and kept for level 2.
    for (int idx = tid; idx < K * deg; idx += BB) {
        const int i = idx / deg, e = idx % deg;
        const int v = mem[i];
        if (v < 0) continue;
        const int x = G[(size_t)v * deg + e];
        if (x < 0) continue;
        for (int j = 0; j < K; ++j)
            if (mem[j] == x) atomicOr(&rch[j], 1u << i);
        if (h > 1) l1_slot(l1key, l1mask, x, 1u << i, overflow);
    }
    __syncthreads();

    // Level 2: expand only the tagged set, and only record where it lands on a
    // member. Masks are read as of the end of level 1, so no path exceeds h.
    if (h > 1) {
        for (int s = 0; s < L1CAP; s += BB / 1) {
            const int slot = s + tid;
            unsigned mask = 0u; int node = -1;
            if (slot < L1CAP && l1key[slot] != 0) { node = l1key[slot] - 1; mask = l1mask[slot]; }
            if (node >= 0 && mask) {
                for (int e = 0; e < deg; ++e) {
                    const int y = G[(size_t)node * deg + e];
                    if (y < 0) continue;
                    for (int j = 0; j < K; ++j)
                        if (mem[j] == y) atomicOr(&rch[j], mask);
                }
            }
        }
    }
    __syncthreads();

    for (int j = tid; j < K; j += BB) reach[(size_t)u * K + j] = rch[j];
}

torch::Tensor neighborhood_reach(torch::Tensor G, torch::Tensor members, int64_t h) {
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == torch::kInt32 && G.is_contiguous(), "G: cuda int32");
    TORCH_CHECK(members.is_cuda() && members.scalar_type() == torch::kInt32 && members.is_contiguous(),
                "members: cuda int32");
    const int N = members.size(0), K = members.size(1), deg = G.size(1);
    TORCH_CHECK(K <= MAXK, "neighbourhood larger than MAXK");
    auto reach = torch::zeros({N, K}, torch::dtype(torch::kInt32).device(G.device()));
    auto overflow = torch::zeros({1}, torch::dtype(torch::kInt32).device(G.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    neighborhood_reach_kernel<<<N, BB, 0, stream>>>(
        G.data_ptr<int>(), members.data_ptr<int>(),
        (unsigned*)reach.data_ptr<int>(), overflow.data_ptr<int>(), N, deg, K, (int)h);
    C10_CUDA_CHECK(cudaGetLastError());
    TORCH_CHECK(overflow.cpu().item<int>() == 0,
                "level-1 set overflowed its table; lower K or raise L1CAP");
    return reach;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("neighborhood_reach", &neighborhood_reach,
            "per-node mutual h-hop reach bitmasks over its own neighbourhood");
}
