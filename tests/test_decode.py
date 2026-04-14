import pytest
import torch

from tests.fixed_point_helpers import requires_cuda
from fxpr_vllm.fixed_point_kernels.decode import (
    decode_attention_fwd_fp_kernel,
)


def _make_decode_inputs(
    batch,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size=1,
    seed=42,
):
    """Build paged-KV inputs for the decode attention kernel."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_seq = max(seq_lens)
    total_pages = sum((s + page_size - 1) // page_size for s in seq_lens)
    total_slots = total_pages * page_size

    k_buffer = (
        torch.randn(
            total_slots,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        * 0.5
    )
    v_buffer = (
        torch.randn(
            total_slots,
            num_kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
            generator=g,
        )
        * 0.5
    )
    q = (
        torch.randn(
            batch, num_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
        )
        * 0.5
    )

    # Build page table: req_to_token[b, token_idx] = global slot
    max_pages_per_seq = (max_seq + page_size - 1) // page_size
    req_to_token = torch.zeros(
        batch, max_pages_per_seq, device="cuda", dtype=torch.int32
    )
    slot_offset = 0
    for b in range(batch):
        n_pages = (seq_lens[b] + page_size - 1) // page_size
        for p in range(n_pages):
            req_to_token[b, p] = slot_offset + p
        slot_offset += n_pages

    b_seq_len = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return q, k_buffer, v_buffer, req_to_token, b_seq_len


def _run_decode_kernel(
    q,
    k_buffer,
    v_buffer,
    req_to_token,
    b_seq_len,
    num_kv_splits=4,
    page_size=1,
):
    """Launch the two-stage decode attention kernel and return output + lse."""
    batch, num_heads, head_dim = q.shape
    Lv = v_buffer.shape[-1]
    o = torch.empty(batch, num_heads, head_dim, device="cuda", dtype=torch.float32)
    lse = torch.empty(batch, num_heads, device="cuda", dtype=torch.float32)
    attn_logits = torch.empty(
        batch,
        num_heads,
        num_kv_splits,
        Lv + 1,
        device="cuda",
        dtype=torch.float32,
    )
    sm_scale = 1.0 / (head_dim**0.5)

    decode_attention_fwd_fp_kernel(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size=page_size,
    )
    return o, lse


def _ref_decode_attention(
    q,
    k_buffer,
    v_buffer,
    req_to_token,
    b_seq_len,
    page_size=1,
):
    """Reference single-query attention using PyTorch, honoring the page table."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group = num_heads // num_kv_heads
    sm_scale = 1.0 / (head_dim**0.5)
    o = torch.zeros_like(q)

    for b in range(batch):
        seq_len = b_seq_len[b].item()
        slots = []
        for t in range(seq_len):
            page_idx = t // page_size
            page_num = req_to_token[b, page_idx].item()
            slot = page_num * page_size + t % page_size
            slots.append(slot)
        slots_t = torch.tensor(slots, device="cuda", dtype=torch.long)

        for h in range(num_heads):
            kv_h = h // kv_group
            qi = q[b, h, :]
            ki = k_buffer[slots_t, kv_h, :]
            vi = v_buffer[slots_t, kv_h, :]

            scores = (ki @ qi) * sm_scale
            p = torch.softmax(scores, dim=0)
            o[b, h, :] = p @ vi
    return o


@requires_cuda
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,num_kv_heads,head_dim",
    [
        (1, 32, 4, 4, 64),
        (4, 64, 8, 4, 64),
        (2, 128, 4, 2, 64),
        (1, 16, 4, 4, 128),
    ],
)
def test_decode_correctness(batch, seq_len, num_heads, num_kv_heads, head_dim):
    """Fixed-point decode attention should closely match a reference implementation."""
    seq_lens = [seq_len] * batch
    q, k_buf, v_buf, req_to_token, b_seq_len = _make_decode_inputs(
        batch,
        seq_lens,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    got, _ = _run_decode_kernel(q, k_buf, v_buf, req_to_token, b_seq_len)
    ref = _ref_decode_attention(q, k_buf, v_buf, req_to_token, b_seq_len)

    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
def test_decode_variable_seq_lens():
    """Batch elements with different sequence lengths should each be correct."""
    seq_lens = [16, 48, 32]
    q, k_buf, v_buf, req_to_token, b_seq_len = _make_decode_inputs(
        3,
        seq_lens,
        4,
        4,
        64,
        seed=99,
    )
    got, _ = _run_decode_kernel(q, k_buf, v_buf, req_to_token, b_seq_len)
    ref = _ref_decode_attention(q, k_buf, v_buf, req_to_token, b_seq_len)

    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
def test_decode_single_token_seq():
    """Sequence of length 1: output should equal V (softmax of single score is 1)."""
    q, k_buf, v_buf, req_to_token, b_seq_len = _make_decode_inputs(
        1,
        [1],
        4,
        4,
        64,
        seed=0,
    )
    got, _ = _run_decode_kernel(
        q, k_buf, v_buf, req_to_token, b_seq_len, num_kv_splits=1
    )
    ref = _ref_decode_attention(q, k_buf, v_buf, req_to_token, b_seq_len)

    assert torch.allclose(got, ref, atol=1e-3), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
def test_decode_paged_kv():
    """Correctness with page_size > 1."""
    page_size = 4
    seq_lens = [32, 16]
    q, k_buf, v_buf, req_to_token, b_seq_len = _make_decode_inputs(
        2,
        seq_lens,
        4,
        4,
        64,
        page_size=page_size,
        seed=77,
    )
    got, _ = _run_decode_kernel(
        q,
        k_buf,
        v_buf,
        req_to_token,
        b_seq_len,
        page_size=page_size,
    )
    ref = _ref_decode_attention(
        q,
        k_buf,
        v_buf,
        req_to_token,
        b_seq_len,
        page_size=page_size,
    )

    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(got - ref).abs().max().item()}"
    )


def _ordered_float_decode_attention_row(
    q_row: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    sm_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scalar single-query attention accumulated left-to-right in `dtype`."""
    seq_len = k_rows.shape[0]
    head_dim = q_row.shape[0]

    qc = q_row.to(dtype)
    kc = k_rows.to(dtype)
    vc = v_rows.to(dtype)

    scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    for j in range(seq_len):
        dot = torch.zeros((), device="cuda", dtype=dtype)
        for d in range(head_dim):
            dot = dot + qc[d] * kc[j, d]
        scores[j] = dot * sm_scale

    scores_max = scores.max()
    exp_scores = torch.zeros(seq_len, device="cuda", dtype=dtype)
    exp_sum = torch.zeros((), device="cuda", dtype=dtype)
    for j in range(seq_len):
        exp_scores[j] = torch.exp(scores[j] - scores_max)
        exp_sum = exp_sum + exp_scores[j]
    p = exp_scores / exp_sum

    out = torch.zeros(head_dim, device="cuda", dtype=dtype)
    for j in range(seq_len):
        for d in range(head_dim):
            out[d] = out[d] + p[j] * vc[j, d]
    return out


def _ordered_fp16_decode_attention_row(
    q_row: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    return _ordered_float_decode_attention_row(
        q_row,
        k_rows,
        v_rows,
        sm_scale,
        torch.float16,
    )


@requires_cuda
def test_fp16_decode_attention_is_order_dependent():
    """Demonstrate that fp16 single-query attention is order-dependent
    when KV rows are permuted."""
    torch.manual_seed(42)
    seq_len, head_dim = 8, 16
    q_row = torch.randn(head_dim, device="cuda", dtype=torch.float32) * 2.0
    k_rows = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32) * 2.0
    v_rows = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32) * 2.0
    # Make one entry large to trigger fp16 rounding
    k_rows[0] *= 20.0
    v_rows[0] *= 20.0
    sm_scale = 1.0 / (head_dim**0.5)

    out_fwd = _ordered_fp16_decode_attention_row(q_row, k_rows, v_rows, sm_scale)

    perm = torch.randperm(seq_len, device="cuda")
    out_perm = _ordered_fp16_decode_attention_row(
        q_row, k_rows[perm], v_rows[perm], sm_scale
    )

    assert not torch.equal(out_fwd, out_perm), (
        "fp16 attention should differ with permuted KV order"
    )


@requires_cuda
def test_decode_kv_permutation_invariance():
    """Fixed-point decode must produce identical output when the KV
    sequence order is permuted."""
    seq_len, num_heads, head_dim = 32, 4, 64
    g = torch.Generator(device="cuda").manual_seed(7)

    k_buffer = (
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
    v_buffer = (
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
    q = (
        torch.randn(
            1, num_heads, head_dim, device="cuda", dtype=torch.float32, generator=g
        )
        * 0.5
    )

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    req_to_token = torch.arange(seq_len, device="cuda", dtype=torch.int32).unsqueeze(0)

    o_orig, _ = _run_decode_kernel(q, k_buffer, v_buffer, req_to_token, b_seq_len)

    perm = torch.randperm(seq_len, device="cuda")
    k_perm = k_buffer[perm]
    v_perm = v_buffer[perm]

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(seq_len, device="cuda", dtype=perm.dtype)
    req_to_token_perm = inv_perm.unsqueeze(0)

    o_perm, _ = _run_decode_kernel(q, k_perm, v_perm, req_to_token_perm, b_seq_len)

    assert torch.allclose(o_orig, o_perm, atol=1e-4, rtol=1e-4), (
        f"Fixed-point decode should be KV-permutation invariant, "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
def test_float_accumulation_is_order_dependent():
    """Demonstrate catastrophic cancellation in fp32 single-query
    attention, and assert the fxp decode kernel is invariant on the
    same inputs.
    """
    seq_len, head_dim = 3, 16

    q_row = torch.ones(head_dim, device="cuda", dtype=torch.float32)
    k_rows = torch.ones(seq_len, head_dim, device="cuda", dtype=torch.float32)

    v_rows = torch.zeros(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v_rows[0] = 1e30
    v_rows[1] = -1e30
    v_rows[2] = 1e-30

    sm_scale = 1.0 / (head_dim**0.5)

    out_fwd = _ordered_float_decode_attention_row(
        q_row,
        k_rows,
        v_rows,
        sm_scale,
        torch.float32,
    )

    perm = torch.tensor([0, 2, 1], device="cuda")
    out_perm = _ordered_float_decode_attention_row(
        q_row,
        k_rows[perm],
        v_rows[perm],
        sm_scale,
        torch.float32,
    )

    assert not torch.equal(out_fwd, out_perm), (
        "fp32 attention should differ with permuted KV order due to "
        "catastrophic cancellation (1e30 + 1e-30 ≠ 1e-30 + 1e30 in fp32)"
    )

    num_heads = 1
    q_kern = q_row.view(1, num_heads, head_dim).contiguous()
    k_buffer = k_rows.unsqueeze(1).contiguous()
    v_buffer = v_rows.unsqueeze(1).contiguous()

    b_seq_len = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    req_to_token = torch.arange(
        seq_len,
        device="cuda",
        dtype=torch.int32,
    ).unsqueeze(0)

    o_orig, _ = _run_decode_kernel(
        q_kern,
        k_buffer,
        v_buffer,
        req_to_token,
        b_seq_len,
    )

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(seq_len, device="cuda", dtype=perm.dtype)
    req_to_token_perm = inv_perm.to(torch.int32).unsqueeze(0)

    o_perm, _ = _run_decode_kernel(
        q_kern,
        k_buffer[perm],
        v_buffer[perm],
        req_to_token_perm,
        b_seq_len,
    )

    assert torch.equal(o_orig, o_perm), (
        f"Fixed-point decode should be KV-permutation invariant even "
        f"with catastrophic-cancellation inputs, "
        f"max diff = {(o_orig - o_perm).abs().max().item()}"
    )


@requires_cuda
def test_decode_deterministic_across_runs():
    """The same inputs must always produce bitwise identical outputs."""
    q, k_buf, v_buf, req_to_token, b_seq_len = _make_decode_inputs(
        2,
        [64, 64],
        4,
        4,
        64,
        seed=77,
    )

    results = []
    for _ in range(5):
        o, _ = _run_decode_kernel(q, k_buf, v_buf, req_to_token, b_seq_len)
        results.append(o)

    for r in results[1:]:
        assert torch.equal(results[0], r), (
            f"Non-deterministic: max diff = {(results[0] - r).abs().max().item()}"
        )
