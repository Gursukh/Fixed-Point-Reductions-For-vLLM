"""torch.ops.fxpr.* schemas and CUDA impls (dispatch to Triton)."""

from __future__ import annotations

import torch

from ._triton import attention as _triton_attention
from ._triton import casts as _triton_casts
from ._triton import gemm as _triton_gemm
from ._triton import rms_norm as _triton_rms_norm


_lib = torch.library.Library("fxpr", "DEF")

_lib.define("float_to_fixed(Tensor x, int int_bits, int fxp_frac_bits) -> Tensor")
_lib.define("fixed_to_float(Tensor x, int float_bits, int fxp_frac_bits) -> Tensor")
# rms_norm reduces in fp32, so no bit args.
_lib.define("rms_norm_fxp(Tensor x, Tensor weight_fp32, float eps) -> Tensor")
_lib.define(
    "rms_norm_fxp_residual(Tensor x, Tensor(a!) residual, Tensor weight_fp32, "
    "float eps) -> Tensor"
)
_lib.define(
    "gemm_fxp(Tensor a, Tensor b, Tensor? bias, int fxp_int_bits, "
    "int fxp_frac_bits) -> Tensor"
)
_lib.define(
    "unified_attention_fxp(Tensor q, Tensor kv_cache, Tensor(a!) o, "
    "Tensor query_start_loc, Tensor seq_lens, Tensor block_table, "
    "int max_query_len, Tensor? alibi_slopes, bool is_causal, "
    "float? softmax_scale, int fxp_int_bits, int fxp_frac_bits, "
    "float logit_softcap, int window_size, int num_kv_splits) -> ()"
)


def _check_frac_bits(n: int) -> None:
    # No range check on purpose: benchmark sweeps need arbitrary frac_bits
    # (even negative or >=64). Kernels derive scale via ldexp, which saturates.
    # int_bits/float_bits stay restricted since they map to real torch dtypes.
    int(n)


def _check_int_bits(n: int) -> None:
    if int(n) not in (16, 32, 64):
        raise ValueError(f"int_bits must be 16/32/64, got {n}")


@torch.library.impl("fxpr::float_to_fixed", "CUDA", lib=_lib)
def _float_to_fixed_cuda(x: torch.Tensor, int_bits: int, fxp_frac_bits: int):
    _check_int_bits(int_bits)
    _check_frac_bits(fxp_frac_bits)
    return _triton_casts.float_to_fixed_run(x, int(int_bits), int(fxp_frac_bits))


@torch.library.impl("fxpr::fixed_to_float", "CUDA", lib=_lib)
def _fixed_to_float_cuda(x: torch.Tensor, float_bits: int, fxp_frac_bits: int):
    if int(float_bits) not in (16, 32, 64):
        raise ValueError(f"float_bits must be 16/32/64, got {float_bits}")
    _check_frac_bits(fxp_frac_bits)
    return _triton_casts.fixed_to_float_run(x, int(float_bits), int(fxp_frac_bits))


@torch.library.impl("fxpr::rms_norm_fxp", "CUDA", lib=_lib)
def _rms_norm_fxp_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float):
    return _triton_rms_norm.rms_norm_fxp_run(x, weight, float(eps))


@torch.library.impl("fxpr::rms_norm_fxp_residual", "CUDA", lib=_lib)
def _rms_norm_fxp_residual_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    return _triton_rms_norm.rms_norm_fxp_residual_run(
        x, residual, weight, float(eps)
    )


@torch.library.impl("fxpr::gemm_fxp", "CUDA", lib=_lib)
def _gemm_fxp_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    fxp_int_bits: int,
    fxp_frac_bits: int,
):
    _check_int_bits(fxp_int_bits)
    _check_frac_bits(fxp_frac_bits)
    return _triton_gemm.gemm_fxp_run(
        a, b, bias, int(fxp_int_bits), int(fxp_frac_bits)
    )


@torch.library.impl("fxpr::unified_attention_fxp", "CUDA", lib=_lib)
def _unified_attention_fxp_cuda(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    o: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_query_len: int,
    alibi_slopes: torch.Tensor | None,
    is_causal: bool,
    softmax_scale: float | None,
    fxp_int_bits: int,
    fxp_frac_bits: int,
    logit_softcap: float,
    window_size: int,
    num_kv_splits: int,
):
    _check_int_bits(fxp_int_bits)
    _check_frac_bits(fxp_frac_bits)
    _triton_attention.unified_attention_fxp_run(
        q,
        kv_cache,
        o,
        query_start_loc,
        seq_lens,
        block_table,
        int(max_query_len),
        alibi_slopes,
        bool(is_causal),
        None if softmax_scale is None else float(softmax_scale),
        int(fxp_int_bits),
        int(fxp_frac_bits),
        float(logit_softcap),
        int(window_size),
        int(num_kv_splits),
    )
