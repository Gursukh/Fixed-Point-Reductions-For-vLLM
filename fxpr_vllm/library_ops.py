"""Public op surface — thin shim over the CUDA extension.

After the CUDA migration, every operation runs through `fxpr_vllm._cuda`,
which registers `torch.ops.fxpr.*` at import time. This module re-exports
those ops under their Python names and provides a small Python helper
for the tier-2 int8 GEMM that bundles per-row scale computation +
quantisation + MMA into one call.
"""

from __future__ import annotations

import torch

from . import _cuda  # noqa: F401  (registers torch.ops.fxpr.*)


# ---------------------------------------------------------------------------
# Fake (meta) implementations for every fxpr::* op.
#
# Without these, torch.compile / dynamo / fx tracers fail with
# "tensor has a non-zero number of elements, but its data is not allocated
#  yet" when they try to dispatch the op against a FakeTensor: the C++
# kernel reads .data_ptr() and crashes. Each fake just allocates an empty
# output tensor with the correct shape and dtype.
# ---------------------------------------------------------------------------


def _int_dtype_for_bits(int_bits: int) -> torch.dtype:
    if int_bits == 16:
        return torch.int16
    if int_bits == 32:
        return torch.int32
    if int_bits == 64:
        return torch.int64
    raise ValueError(f"int_bits must be 16/32/64, got {int_bits}")


def _float_dtype_for_bits(float_bits: int) -> torch.dtype:
    if float_bits == 16:
        return torch.float16
    if float_bits == 32:
        return torch.float32
    if float_bits == 64:
        return torch.float64
    raise ValueError(f"float_bits must be 16/32/64, got {float_bits}")


@torch.library.register_fake("fxpr::float_to_fixed")
def _float_to_fixed_fake(x, frac_bits, int_bits):
    return torch.empty_like(x, dtype=_int_dtype_for_bits(int(int_bits)))


@torch.library.register_fake("fxpr::fixed_to_float")
def _fixed_to_float_fake(x, frac_bits, float_bits):
    return torch.empty_like(x, dtype=_float_dtype_for_bits(int(float_bits)))


@torch.library.register_fake("fxpr::rms_norm_fxp")
def _rms_norm_fxp_fake(x, weight_fp32, eps, frac_bits, fxp_int_bits):
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::rms_norm_fxp_residual")
def _rms_norm_fxp_residual_fake(x, residual, weight_fp32, eps, frac_bits, fxp_int_bits):
    # residual is mutated in place; the fake follows the schema's (a!) marker.
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::log_softmax_fxp")
def _log_softmax_fxp_fake(x, frac_bits, fxp_int_bits):
    return torch.empty_like(x)


@torch.library.register_fake("fxpr::compute_per_row_scale")
def _compute_per_row_scale_fake(x, eps):
    out_shape = list(x.shape[:-1])
    return x.new_empty(out_shape, dtype=torch.float16)


@torch.library.register_fake("fxpr::gemm_fxp")
def _gemm_fxp_fake(a, b, frac_bits, fxp_int_bits):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


@torch.library.register_fake("fxpr::gemm_fxp_int8")
def _gemm_fxp_int8_fake(a_int8, a_scale, b_int8, b_scale, frac_bits, fxp_int_bits):
    return a_int8.new_empty(
        (a_int8.shape[0], b_int8.shape[1]), dtype=torch.float32
    )


@torch.library.register_fake("fxpr::unified_attention_fxp")
def _unified_attention_fxp_fake(
    q,
    kv_cache,
    o,
    query_start_loc,
    seq_lens,
    block_table,
    max_query_len,
    alibi_slopes,
    is_causal,
    softmax_scale,
    frac_bits,
    fxp_int_bits,
    logit_softcap,
    window_size,
):
    # The op writes into `o` in-place and returns None.
    return None


# Re-exports of the registered torch.ops.fxpr.* ops.
gemm_fxp = torch.ops.fxpr.gemm_fxp
gemm_fxp_int8 = torch.ops.fxpr.gemm_fxp_int8
rms_norm_fxp = torch.ops.fxpr.rms_norm_fxp
rms_norm_fxp_residual = torch.ops.fxpr.rms_norm_fxp_residual
log_softmax_fxp = torch.ops.fxpr.log_softmax_fxp
compute_per_row_scale = torch.ops.fxpr.compute_per_row_scale
float_to_fixed = torch.ops.fxpr.float_to_fixed
fixed_to_float = torch.ops.fxpr.fixed_to_float


def quantise_to_int8(
    x: torch.Tensor, scale_fp16: torch.Tensor
) -> torch.Tensor:
    """Per-row int8 quantisation: x_int8[i, k] = clamp(round(x[i,k] / s[i]), [-127, 127]).

    Both x and scale_fp16 are expected on the same device.
    The scale tensor must be precomputed via :func:`compute_per_row_scale`
    (or any equivalently split-invariant procedure) so the result stays
    bit-identical regardless of how rows are partitioned across launches.
    """
    assert x.is_cuda and scale_fp16.is_cuda
    assert x.dim() == 2
    assert scale_fp16.shape == (x.shape[0],)
    s = scale_fp16.to(torch.float32).unsqueeze(-1)
    q = torch.round(x / s).clamp_(-127, 127).to(torch.int8)
    return q


def launch_gemm_fxp_mma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    q_frac_bits: int = 8,
    frac_bits: int = 16,
    fxp_int_bits: int = 32,
    block_k: int | None = None,
) -> torch.Tensor:
    """Tier-2 int8 GEMM with per-row/per-col scales.

    The original Triton-era signature took q_frac_bits to drive
    static int8 scaling. The deterministic CUDA path computes
    split-invariant per-row scales instead; q_frac_bits and
    block_k are accepted for backwards compatibility but ignored.
    The K-accumulator is integer over int8 products, so the result is
    bit-identical for any K-block schedule.
    """
    del q_frac_bits, block_k

    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]

    a_c = a.contiguous()
    b_t = b.t().contiguous()  # per-col scales of B = per-row scales of B.T

    a_scale = compute_per_row_scale(a_c, 1e-8)
    b_scale = compute_per_row_scale(b_t, 1e-8)

    a_int8 = quantise_to_int8(a_c, a_scale)
    b_t_int8 = quantise_to_int8(b_t, b_scale)
    b_int8 = b_t_int8.t().contiguous()

    return gemm_fxp_int8(a_int8, a_scale, b_int8, b_scale, frac_bits, fxp_int_bits)
