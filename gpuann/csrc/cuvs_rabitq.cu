// A thin binding to cuVS's own GPU IVF-RaBitQ, for use as a baseline.
//
// cuVS 26.10 ships cuvs::neighbors::ivf_rabitq -- the extended multi-bit
// variant, `bits_per_dim` defaulting to 3, with LUT and quantised search modes
// -- but exposes it neither through its C API nor its Python package. Rather
// than compare against our own one-bit implementation and call it "RaBitQ on a
// GPU", this links the library directly so the baseline is the authors' code.
//
// Build it against the wheel's headers; see bench/cuvs_rabitq.py for the flags.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>

#include <memory>

namespace {

struct Index {
  raft::device_resources handle;
  std::unique_ptr<cuvs::neighbors::ivf_rabitq::index<int64_t>> idx;
};

std::vector<std::shared_ptr<Index>> g_indexes;

}  // namespace

int64_t rabitq_build(torch::Tensor X, int64_t n_lists, int64_t bits_per_dim,
                     int64_t kmeans_iters, bool inner_product) {
  TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32 && X.is_contiguous(),
              "dataset must be a contiguous cuda float32 matrix");
  auto h = std::make_shared<Index>();
  cuvs::neighbors::ivf_rabitq::index_params p;
  p.n_lists = static_cast<uint32_t>(n_lists);
  p.bits_per_dim = static_cast<uint32_t>(bits_per_dim);
  p.kmeans_n_iters = static_cast<uint32_t>(kmeans_iters);
  p.metric = inner_product ? cuvs::distance::DistanceType::InnerProduct
                           : cuvs::distance::DistanceType::L2Expanded;
  auto view = raft::make_device_matrix_view<const float, int64_t>(
      X.data_ptr<float>(), X.size(0), X.size(1));
  h->idx = std::make_unique<cuvs::neighbors::ivf_rabitq::index<int64_t>>(
      cuvs::neighbors::ivf_rabitq::build(h->handle, p, view));
  h->handle.sync_stream();
  g_indexes.push_back(h);
  return static_cast<int64_t>(g_indexes.size() - 1);
}

std::vector<torch::Tensor> rabitq_search(int64_t id, torch::Tensor Q, int64_t k,
                                         int64_t n_probes, int64_t mode) {
  TORCH_CHECK(id >= 0 && id < (int64_t)g_indexes.size(), "no such index");
  TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kFloat32 && Q.is_contiguous(),
              "queries must be a contiguous cuda float32 matrix");
  auto& h = *g_indexes[id];
  cuvs::neighbors::ivf_rabitq::search_params sp;
  sp.n_probes = static_cast<uint32_t>(n_probes);
  sp.mode = static_cast<cuvs::neighbors::ivf_rabitq::search_mode>(mode);

  auto nb = torch::empty({Q.size(0), k},
                         torch::dtype(torch::kInt64).device(Q.device()));
  auto ds = torch::empty({Q.size(0), k},
                         torch::dtype(torch::kFloat32).device(Q.device()));
  auto qv = raft::make_device_matrix_view<const float, int64_t>(
      Q.data_ptr<float>(), Q.size(0), Q.size(1));
  auto nv = raft::make_device_matrix_view<int64_t, int64_t>(
      nb.data_ptr<int64_t>(), Q.size(0), k);
  auto dv = raft::make_device_matrix_view<float, int64_t>(
      ds.data_ptr<float>(), Q.size(0), k);
  cuvs::neighbors::ivf_rabitq::search(h.handle, sp, *h.idx, qv, nv, dv);
  h.handle.sync_stream();
  return {nb, ds};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build", &rabitq_build, "build a cuVS IVF-RaBitQ index");
  m.def("search", &rabitq_search, "search a cuVS IVF-RaBitQ index");
}
