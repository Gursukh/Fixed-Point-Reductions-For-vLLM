import pytest
import torch
import triton
import triton.language as tl

from fxpr_vllm.fixed_point_kernels.fixed_point import (
    float_to_fixed as _float_to_fixed_jit,
    fixed_to_float as _fixed_to_float_jit,
)


@triton.jit
def _float_to_fixed_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = _float_to_fixed_jit(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _fixed_to_float_kernel(
    x_ptr, y_ptr, n, frac_bits: tl.constexpr, BLOCK: tl.constexpr, OUT: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = _fixed_to_float_jit(x, frac_bits, OUT)
    tl.store(y_ptr + offs, y, mask=mask)


_TL = {
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def float_to_fixed(
    x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype
) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _float_to_fixed_kernel[(1,)](
        x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out]
    )
    return y.view_as(x)


def fixed_to_float(
    x: torch.Tensor, frac_bits: tl.constexpr, out: torch.dtype
) -> torch.Tensor:
    n = x.numel()
    block = triton.next_power_of_2(max(n, 1))
    y = torch.empty(n, device=x.device, dtype=out)
    _fixed_to_float_kernel[(1,)](
        x.contiguous(), y, n, frac_bits, BLOCK=block, OUT=_TL[out]
    )
    return y.view_as(x)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gemm_fxp_test(
    a: torch.Tensor, b: torch.Tensor, frac_bits: int = 16, fxp_int_bits: int = 32
) -> torch.Tensor:
    """Test adapter for the registered :func:`fxpr::gemm_fxp` torch op."""
    from fxpr_vllm.library_ops import gemm_fxp as gemm_fxp_op

    return gemm_fxp_op(a.contiguous(), b.contiguous(), frac_bits, fxp_int_bits)


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
    frac_bits: int = 14,
) -> None:
    """Test adapter that runs prefill via the unified paged-KV attention kernel.

    Builds a synthetic page_size=1 KV cache from packed (seq, head, dim) k/v.
    Each token occupies its own physical block, so the page-table lookup
    becomes the absolute token index and the kernel's paged path collapses
    to the non-paged behaviour the prefill tests originally exercised.
    """
    import triton.language as tl

    from fxpr_vllm.fixed_point_kernels.attention import unified_attention_fxp
    from fxpr_vllm.fixed_point_kernels.fixed_point import RCP_LN2

    # unified_attention_fxp expects pre-scaled (by RCP_LN2) alibi slopes.
    if alibi_slopes is not None:
        alibi_slopes = alibi_slopes * RCP_LN2

    total_tokens, num_kv_heads, head_dim = k.shape

    # page_size=1 KV cache: shape (total_tokens, 2, 1, num_kv_heads, head_dim).
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

    # block_table: each request maps logical block i -> absolute token index.
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

    # query_start_loc is cumulative with a trailing total: shape (n+1,).
    query_start_loc = torch.empty(num_requests + 1, device=q.device, dtype=torch.int32)
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.cumsum(b_seq_len, dim=0)

    unified_attention_fxp(
        q=q,
        kv_cache=kv_cache,
        o=o,
        query_start_loc=query_start_loc,
        seq_lens=b_seq_len.to(torch.int32),
        block_table=block_table,
        max_query_len=max_input_len,
        alibi_slopes=alibi_slopes,
        is_causal=is_causal,
        softmax_scale=softmax_scale,
        frac_bits=frac_bits,
        fxp_dtype=tl.int32,
    )
