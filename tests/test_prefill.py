import pytest
import torch

from tests.fixed_point_helpers import requires_cuda
from fxpr_vllm.fixed_point_kernels.prefill import (
    context_attention_fwd_fxp_kernel,
)


def _ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference multi-head attention computed per-sequence with PyTorch."""
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group = num_heads // num_kv_heads
    o = torch.zeros_like(q)

    for b in range(b_seq_len.shape[0]):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()
        for h in range(num_heads):
            kv_h = h // kv_group
            qi = q[start : start + seq_len, h, :]
            ki = k[start : start + seq_len, kv_h, :]
            vi = v[start : start + seq_len, kv_h, :]

            scores = (qi @ ki.T) * sm_scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores.masked_fill_(mask, -1e8)
            p = torch.softmax(scores, dim=-1)
            o[start : start + seq_len, h, :] = p @ vi
    return o


def _make_inputs(batch, seq_lens, num_heads, num_kv_heads, head_dim, seed=42):
    """Build packed QKV tensors and metadata for prefill."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    q = torch.randn(
        total, num_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    k = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )
    v = torch.randn(
        total, num_kv_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
    )

    b_seq_len = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    b_start_loc = torch.zeros(batch, device="cuda", dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], dim=0)

    o = torch.empty_like(q)
    return q, k, v, o, b_start_loc, b_seq_len


@requires_cuda
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,num_kv_heads,head_dim",
    [
        (1, 8, 1, 1, 16),
        (2, 32, 4, 2, 32),
        (1, 64, 4, 4, 64),
    ],
)
def test_prefill_correctness_causal(batch, seq_len, num_heads, num_kv_heads, head_dim):
    """Fixed-point prefill attention should closely match a reference implementation (causal)."""
    seq_lens = [seq_len] * batch
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(
        batch, seq_lens, num_heads, num_kv_heads, head_dim
    )
    sm_scale = 1.0 / (head_dim**0.5)

    context_attention_fwd_fxp_kernel(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=True,
        softmax_scale=sm_scale,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=sm_scale
    )

    assert torch.allclose(o, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(o - ref).abs().max().item()}"
    )


@requires_cuda
def test_prefill_correctness_non_causal():
    """Fixed-point prefill attention should closely match reference (non-causal)."""
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(1, [32], 2, 2, 32)
    sm_scale = 1.0 / (32**0.5)

    context_attention_fwd_fxp_kernel(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=32,
        is_causal=False,
        softmax_scale=sm_scale,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=False, sm_scale=sm_scale
    )

    assert torch.allclose(o, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(o - ref).abs().max().item()}"
    )


@requires_cuda
def test_prefill_variable_seq_lens():
    """Batches with different sequence lengths should each be correct."""
    seq_lens = [16, 32, 24]
    q, k, v, o, b_start_loc, b_seq_len = _make_inputs(3, seq_lens, 2, 2, 32, seed=99)
    sm_scale = 1.0 / (32**0.5)

    context_attention_fwd_fxp_kernel(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len=max(seq_lens),
        is_causal=True,
        softmax_scale=sm_scale,
    )
    ref = _ref_attention(
        q, k, v, b_start_loc, b_seq_len, is_causal=True, sm_scale=sm_scale
    )

    assert torch.allclose(o, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(o - ref).abs().max().item()}"
    )


def _float_attention_row(
    q_row: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    sm_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scalar attention for a single query, accumulated left-to-right in *dtype*."""
    seq_len = k_rows.shape[0]
    head_dim = q_row.shape[0]

    q_c = q_row.to(dtype)
    k_c = k_rows.to(dtype)

    # Dot products accumulated one element at a time
    scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    for j in range(seq_len):
        dot = torch.zeros((), device="cuda", dtype=dtype)
        for d in range(head_dim):
            dot = dot + q_c[d] * k_c[j, d]
        scores[j] = dot * sm_scale

    # Softmax in the same dtype
    scores_max = scores.max()
    exp_scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    exp_sum = torch.zeros((), device="cuda", dtype=dtype)
    for j in range(seq_len):
        exp_scores[j] = torch.exp(scores[j] - scores_max)
        exp_sum = exp_sum + exp_scores[j]
    p = exp_scores / exp_sum

    # Weighted sum of V rows
    v_c = v_rows.to(dtype)
    out = torch.zeros(head_dim, device="cuda", dtype=dtype)
    for j in range(seq_len):
        for d in range(head_dim):
            out[d] = out[d] + p[j] * v_c[j, d]
    return out


@requires_cuda
def test_float_accumulation_is_order_dependent():
    """Demonstrate catastrophic cancellation in fp32 sequential accumulation."""
    seq_len, head_dim = 3, 16

    # Q and K chosen so attention weights are roughly uniform
    q_row = torch.ones(head_dim, device="cuda", dtype=torch.float32)
    k_rows = torch.ones(seq_len, head_dim, device="cuda", dtype=torch.float32)

    # V with catastrophic-cancellation magnitudes
    v_rows = torch.zeros(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v_rows[0] = 1e30  # large positive
    v_rows[1] = -1e30  # large negative (cancels row 0)
    v_rows[2] = 1e-30  # tiny — swallowed when added after a large value

    sm_scale = 1.0 / (head_dim**0.5)

    out_fwd = _float_attention_row(q_row, k_rows, v_rows, sm_scale, torch.float32)

    # Permutation that interleaves large and tiny values
    perm = torch.tensor([0, 2, 1], device="cuda")
    out_perm = _float_attention_row(
        q_row,
        k_rows[perm],
        v_rows[perm],
        sm_scale,
        torch.float32,
    )

    assert not torch.equal(out_fwd[perm], out_perm), (
        "fp32 attention should differ with permuted KV order due to "
        "catastrophic cancellation (1e30 + 1e-30 ≠ 1e-30 + 1e30 in fp32)"
    )

    num_heads = 1
    q_kern = (
        q_row.unsqueeze(0)
        .unsqueeze(0)
        .expand(seq_len, num_heads, head_dim)
        .contiguous()
    )
    k_kern = k_rows.unsqueeze(1).expand(seq_len, num_heads, head_dim).contiguous()
    v_kern = v_rows.unsqueeze(1).expand(seq_len, num_heads, head_dim).contiguous()

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)

    o_orig = torch.empty_like(q_kern)
    context_attention_fwd_fxp_kernel(
        q_kern,
        k_kern,
        v_kern,
        o_orig,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=sm_scale,
    )

    o_perm = torch.empty_like(q_kern)
    context_attention_fwd_fxp_kernel(
        q_kern,
        k_kern[perm],
        v_kern[perm],
        o_perm,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=sm_scale,
    )

    assert torch.equal(o_orig[perm], o_perm), (
        f"Fixed-point prefill should be KV-permutation invariant, "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
def test_fixedpoint_prefill_is_permutation_equivariant():
    """Fixed-point prefill must produce the same output regardless of KV order."""
    seq_len, num_heads, head_dim = 32, 2, 32
    g = torch.Generator(device="cuda").manual_seed(7)
    q = (
        torch.randn(
            seq_len,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        * 0.5
    )
    k = (
        torch.randn(
            seq_len,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        * 0.5
    )
    v = (
        torch.randn(
            seq_len,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        * 0.5
    )

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    b_start_loc = torch.tensor([0], device="cuda", dtype=torch.int32)
    sm_scale = 1.0 / (head_dim**0.5)

    o_orig = torch.empty_like(q)
    context_attention_fwd_fxp_kernel(
        q,
        k,
        v,
        o_orig,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=sm_scale,
    )

    perm = torch.randperm(seq_len, device="cuda")
    o_perm = torch.empty_like(q)
    context_attention_fwd_fxp_kernel(
        q,
        k[perm],
        v[perm],
        o_perm,
        b_start_loc,
        b_seq_len,
        max_input_len=seq_len,
        is_causal=False,
        softmax_scale=sm_scale,
    )

    assert torch.equal(o_orig, o_perm), (
        f"Fixed-point prefill should be KV-permutation invariant, "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
def test_prefill_deterministic_across_runs():
    """The same inputs must always produce bitwise identical outputs."""
    q, k, v, _, b_start_loc, b_seq_len = _make_inputs(1, [64], 2, 2, 32, seed=77)
    sm_scale = 1.0 / (32**0.5)

    results = []
    for _ in range(5):
        o = torch.empty_like(q)
        context_attention_fwd_fxp_kernel(
            q,
            k,
            v,
            o,
            b_start_loc,
            b_seq_len,
            max_input_len=64,
            is_causal=True,
            softmax_scale=sm_scale,
        )
        results.append(o)

    for r in results[1:]:
        assert torch.equal(results[0], r), (
            f"Non-deterministic: max diff = {(results[0] - r).abs().max().item()}"
        )
