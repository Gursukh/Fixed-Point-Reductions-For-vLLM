// Top-level torch.library registrations for the fxpr namespace.
//
// Schema and CUDA implementation are registered separately:
//   * TORCH_LIBRARY(fxpr, m)       — declares schemas (Tensor in, Tensor out).
//   * TORCH_LIBRARY_IMPL(fxpr, CUDA, m) — binds each schema's CUDA backend.
//
// This split is required so that fxpr_vllm/library_ops.py can register
// fake (meta) implementations via torch.library.register_fake. If we
// used the m.def("schema", &fn) shortcut instead, the implementation
// would land under the CompositeImplicitAutograd dispatch key and
// register_fake would refuse with "the operator already has an
// implementation for this device type via a pre-existing registration
// to DispatchKey::CompositeImplicitAutograd".

#include <torch/library.h>
#include <torch/extension.h>

namespace fxpr {

torch::Tensor float_to_fixed_op(torch::Tensor x, int64_t frac_bits,
                                int64_t int_bits);
torch::Tensor fixed_to_float_op(torch::Tensor x, int64_t frac_bits,
                                int64_t float_bits);
torch::Tensor rms_norm_fxp_op(torch::Tensor x, torch::Tensor w, double eps,
                              int64_t frac_bits, int64_t int_bits);
torch::Tensor rms_norm_fxp_residual_op(torch::Tensor x, torch::Tensor residual,
                                       torch::Tensor w, double eps,
                                       int64_t frac_bits, int64_t int_bits);
torch::Tensor log_softmax_fxp_op(torch::Tensor x, int64_t frac_bits,
                                 int64_t int_bits);
torch::Tensor compute_per_row_scale_op(torch::Tensor x, double eps);
torch::Tensor gemm_fxp_op(torch::Tensor a, torch::Tensor b,
                          int64_t frac_bits, int64_t int_bits);
torch::Tensor gemm_fxp_int8_op(torch::Tensor a_int8, torch::Tensor a_scale,
                               torch::Tensor b_int8, torch::Tensor b_scale,
                               int64_t frac_bits, int64_t int_bits);
void unified_attention_fxp_op(torch::Tensor q, torch::Tensor kv_cache,
                              torch::Tensor o, torch::Tensor query_start_loc,
                              torch::Tensor seq_lens, torch::Tensor block_table,
                              int64_t max_query_len,
                              c10::optional<torch::Tensor> alibi_slopes,
                              bool is_causal,
                              c10::optional<double> softmax_scale,
                              int64_t frac_bits, int64_t fxp_int_bits,
                              double logit_softcap, int64_t window_size);

}  // namespace fxpr

TORCH_LIBRARY(fxpr, m) {
  m.def("float_to_fixed(Tensor x, int frac_bits, int int_bits) -> Tensor");
  m.def("fixed_to_float(Tensor x, int frac_bits, int float_bits) -> Tensor");
  m.def(
      "rms_norm_fxp(Tensor x, Tensor weight_fp32, float eps, int frac_bits, "
      "int fxp_int_bits) -> Tensor");
  m.def(
      "rms_norm_fxp_residual(Tensor x, Tensor(a!) residual, Tensor weight_fp32, "
      "float eps, int frac_bits, int fxp_int_bits) -> Tensor");
  m.def(
      "log_softmax_fxp(Tensor x, int frac_bits, int fxp_int_bits) -> Tensor");
  m.def(
      "compute_per_row_scale(Tensor x, float eps) -> Tensor");
  m.def(
      "gemm_fxp(Tensor a, Tensor b, int frac_bits, int fxp_int_bits) -> Tensor");
  m.def(
      "gemm_fxp_int8(Tensor a_int8, Tensor a_scale, Tensor b_int8, "
      "Tensor b_scale, int frac_bits, int fxp_int_bits) -> Tensor");
  m.def(
      "unified_attention_fxp(Tensor q, Tensor kv_cache, Tensor(a!) o, "
      "Tensor query_start_loc, Tensor seq_lens, Tensor block_table, "
      "int max_query_len, Tensor? alibi_slopes, bool is_causal, "
      "float? softmax_scale, int frac_bits, int fxp_int_bits, "
      "float logit_softcap, int window_size) -> ()");
}

TORCH_LIBRARY_IMPL(fxpr, CUDA, m) {
  m.impl("float_to_fixed", &fxpr::float_to_fixed_op);
  m.impl("fixed_to_float", &fxpr::fixed_to_float_op);
  m.impl("rms_norm_fxp", &fxpr::rms_norm_fxp_op);
  m.impl("rms_norm_fxp_residual", &fxpr::rms_norm_fxp_residual_op);
  m.impl("log_softmax_fxp", &fxpr::log_softmax_fxp_op);
  m.impl("compute_per_row_scale", &fxpr::compute_per_row_scale_op);
  m.impl("gemm_fxp", &fxpr::gemm_fxp_op);
  m.impl("gemm_fxp_int8", &fxpr::gemm_fxp_int8_op);
  m.impl("unified_attention_fxp", &fxpr::unified_attention_fxp_op);
}

// Empty pybind module so `import fxpr_vllm._cuda` succeeds. The real
// op surface is exposed via torch.ops.fxpr.*; this module exists only
// to give CPython something to import (and to load the .so so the
// TORCH_LIBRARY constructor above runs).
PYBIND11_MODULE(_cuda, m) {
  m.doc() = "fxpr_vllm CUDA kernels (registered via torch.ops.fxpr)";
}
