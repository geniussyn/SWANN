// RaBitQ scan: estimate inner products straight out of the packed sign codes.
//
// For a vector x in list l with centroid c, RaBitQ keeps sign(R(x - c)) as D
// bits, ||x - c||, and <o_bar, o_res> -- the cosine between the sign vector and
// the true residual direction, which is what makes the estimate unbiased. Then
//
//     <q, x>  ~=  <q, c>  +  ||x-c|| * ( <Rq, o_bar> / err )
//
// Two things make this cheap. The rotation is orthogonal, so Rq does not depend
// on which list a candidate is in and is computed once per query rather than
// once per probe. And <Rq, o_bar> needs no multiplies at all: with o_bar = +-1
// over sqrt(D),
//
//     <Rq, o_bar> = ( 2 * sum_{bits set} Rq_i  -  sum_i Rq_i ) / sqrt(D)
//
// so a nibble table of partial sums turns the scan into D/4 shared-memory
// lookups over D/8 bytes of code, against D multiply-adds over 4D bytes for the
// exact distance. The query is never quantised, which is the asymmetry the
// one-bit scheme needs to be worth anything.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define RB_BLOCK 256
#define MAXLUT   (1024 / 4 * 16)      // D <= 1024 -> 4096 floats = 16 KiB

__global__ void rabitq_scan_kernel(const unsigned* __restrict__ codes,
                                   const float* __restrict__ rnorm,
                                   const float* __restrict__ err,
                                   const float* __restrict__ qrot,
                                   const int*   __restrict__ cand,
                                   const float* __restrict__ cdot,
                                   float* __restrict__ out,
                                   int D, int W, int C) {
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float lut[];          // (D/4) * 16

    const int nib = D / 4;
    const bool use_lut = (D <= 1024);     // the table is 16 KiB at D = 1024
    const float* qr = qrot + (size_t)q * D;

    // lut[j*16 + v] = sum of qr over the bits set in nibble value v at nibble j
    for (int j = tid; use_lut && j < nib; j += RB_BLOCK) {
        const float a = qr[4 * j], b = qr[4 * j + 1],
                    c = qr[4 * j + 2], d = qr[4 * j + 3];
        float* t = lut + j * 16;
        t[0]  = 0.f;      t[1]  = a;        t[2]  = b;        t[3]  = a + b;
        t[4]  = c;        t[5]  = a + c;    t[6]  = b + c;    t[7]  = a + b + c;
        t[8]  = d;        t[9]  = a + d;    t[10] = b + d;    t[11] = a + b + d;
        t[12] = c + d;    t[13] = a + c + d; t[14] = b + c + d; t[15] = a + b + c + d;
    }
    // Wide vectors keep the query itself in shared instead: slower per
    // candidate than the table, but the table would not fit.
    float* sq = lut;
    for (int t = tid; !use_lut && t < D; t += RB_BLOCK) sq[t] = qr[t];
    __shared__ float qtot;
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += qr[i];
        qtot = s;
    }
    __syncthreads();

    const float inv_sqrt_d = rsqrtf((float)D);
    for (int ci = tid; ci < C; ci += RB_BLOCK) {
        const int v = cand[(size_t)q * C + ci];
        if (v < 0) { out[(size_t)q * C + ci] = -3.4e38f; continue; }
        const unsigned* code = codes + (size_t)v * W;
        float acc = 0.f;
        if (use_lut) {
            #pragma unroll 4
            for (int w = 0; w < W; ++w) {
                unsigned word = code[w];
                const int base = w * 8;              // 8 nibbles per 32-bit word
                #pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
                    acc += lut[(base + nb) * 16 + (word & 0xF)];
                    word >>= 4;
                }
            }
        } else {
            for (int w = 0; w < W; ++w) {
                unsigned word = code[w];
                const int base = w * 32;
                while (word) {
                    const int b = __ffs(word) - 1;
                    acc += sq[base + b];
                    word &= word - 1;
                }
            }
        }
        const float ip_res = (2.f * acc - qtot) * inv_sqrt_d;
        out[(size_t)q * C + ci] = cdot[(size_t)q * C + ci] + rnorm[v] * ip_res / err[v];
    }
}

torch::Tensor rabitq_scan(torch::Tensor codes, torch::Tensor rnorm, torch::Tensor err,
                          torch::Tensor qrot, torch::Tensor cand, torch::Tensor cdot) {
    TORCH_CHECK(codes.is_cuda() && codes.scalar_type() == torch::kInt32 && codes.is_contiguous());
    TORCH_CHECK(qrot.is_cuda() && qrot.scalar_type() == torch::kFloat32 && qrot.is_contiguous());
    TORCH_CHECK(cand.scalar_type() == torch::kInt32 && cand.is_contiguous());
    const int Q = qrot.size(0), D = qrot.size(1), W = codes.size(1), C = cand.size(1);
    TORCH_CHECK(D % 32 == 0, "D must be a multiple of 32");
    auto out = torch::empty({Q, C}, torch::dtype(torch::kFloat32).device(codes.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shmem = (D <= 1024 ? (size_t)D / 4 * 16 : (size_t)D) * sizeof(float);
    rabitq_scan_kernel<<<Q, RB_BLOCK, shmem, stream>>>(
        (const unsigned*)codes.data_ptr<int>(), rnorm.data_ptr<float>(),
        err.data_ptr<float>(), qrot.data_ptr<float>(), cand.data_ptr<int>(),
        cdot.data_ptr<float>(), out.data_ptr<float>(), D, W, C);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("rabitq_scan", &rabitq_scan, "estimate inner products from packed sign codes");
}
