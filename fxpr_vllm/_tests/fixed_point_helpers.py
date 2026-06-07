import pytest
import torch

import fxpr_vllm  # noqa: F401

_INT_BITS_BY_DTYPE = {
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
}

_FLOAT_BITS_BY_DTYPE = {
    torch.float16: 16,
    torch.float32: 32,
    torch.float64: 64,
}

DEFAULT_FXP_FRAC_BITS = 16


def float_to_fixed(
    x: torch.Tensor,
    out: torch.dtype,
    fxp_frac_bits: int = DEFAULT_FXP_FRAC_BITS,
) -> torch.Tensor:
    int_bits = _INT_BITS_BY_DTYPE[out]
    return torch.ops.fxpr.float_to_fixed(x.contiguous(), int_bits, int(fxp_frac_bits))


def fixed_to_float(
    x: torch.Tensor,
    out: torch.dtype,
    fxp_frac_bits: int = DEFAULT_FXP_FRAC_BITS,
) -> torch.Tensor:
    float_bits = _FLOAT_BITS_BY_DTYPE[out]
    return torch.ops.fxpr.fixed_to_float(x.contiguous(), float_bits, int(fxp_frac_bits))


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gemm_dtype_supported(dtype: torch.dtype) -> bool:
    """fp16 needs sm_75 or newer, bf16/fp32 need sm_80 or newer."""
    if not torch.cuda.is_available():
        return False
    cap_major, cap_minor = torch.cuda.get_device_capability()
    cap = cap_major * 10 + cap_minor
    if dtype == torch.float16:
        return cap >= 75
    if dtype in (torch.bfloat16, torch.float32):
        return cap >= 80
    return False


def skip_if_dtype_unsupported(dtype: torch.dtype) -> None:
    if not gemm_dtype_supported(dtype):
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        pytest.skip(f"gemm_fxp does not support {dtype} on sm_{cap[0]}{cap[1]}")


def gemm_fxp_test(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    fxp_int_bits: int = 32,
    fxp_frac_bits: int = DEFAULT_FXP_FRAC_BITS,
) -> torch.Tensor:
    return torch.ops.fxpr.gemm_fxp(
        a.contiguous(),
        b.contiguous(),
        bias if bias is None else bias.contiguous(),
        int(fxp_int_bits),
        int(fxp_frac_bits),
    )


RCP_LN2 = 1.4426950408889634


def prefill_fxp_test(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    *,
    alibi_slopes: torch.Tensor | None = None,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    num_kv_splits: int = 1,
    fxp_int_bits: int = 32,
    fxp_frac_bits: int = DEFAULT_FXP_FRAC_BITS,
) -> None:
    # page_size 1, so one block per token.
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes * RCP_LN2

    total_tokens, num_kv_heads, head_dim = k.shape

    kv_cache = torch.empty(
        total_tokens,
        2,
        1,
        num_kv_heads,
        head_dim,
        device=k.device,
        dtype=k.dtype,
    )
    kv_cache[:, 0, 0] = k
    kv_cache[:, 1, 0] = v

    num_requests = int(b_seq_len.shape[0])
    block_table = torch.zeros(
        num_requests, max_input_len, device=q.device, dtype=torch.int32
    )
    seq_lens_list = b_seq_len.tolist()
    starts_list = b_start_loc.tolist()
    for r, (start, length) in enumerate(zip(starts_list, seq_lens_list)):
        block_table[r, :length] = torch.arange(
            start, start + length, device=q.device, dtype=torch.int32
        )

    query_start_loc = torch.empty(num_requests + 1, device=q.device, dtype=torch.int32)
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.cumsum(b_seq_len, dim=0)

    torch.ops.fxpr.unified_attention_fxp(
        q,
        kv_cache,
        o,
        query_start_loc,
        b_seq_len.to(torch.int32),
        block_table,
        int(max_input_len),
        alibi_slopes,
        bool(is_causal),
        None if softmax_scale is None else float(softmax_scale),
        int(fxp_int_bits),
        int(fxp_frac_bits),
        0.0,
        0,
        int(num_kv_splits),
    )
