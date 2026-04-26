import pytest
import torch

from tests.fixed_point_helpers import gemm_fxp_test, requires_cuda


def _ordered_fp16_dot(
    a_rows: list[list[float]], b_cols: list[list[float]]
) -> torch.Tensor:
    """Scalar fp16 dot-products accumulated left-to-right, to expose ordering effects."""
    M = len(a_rows)
    N = len(b_cols)
    out = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    for i, row in enumerate(a_rows):
        for j, col in enumerate(b_cols):
            acc = torch.zeros((), device="cuda", dtype=torch.float16)
            for a_val, b_val in zip(row, col):
                a_t = torch.tensor(a_val, device="cuda", dtype=torch.float16)
                b_t = torch.tensor(b_val, device="cuda", dtype=torch.float16)
                acc = acc + a_t * b_t
            out[i, j] = acc
    return out


@requires_cuda
@pytest.mark.parametrize("M,N,K", [(4, 4, 16), (3, 7, 16), (1, 1, 32), (8, 16, 32)])
def test_gemm_fixed_point_correctness(M, N, K):
    """Fixed-point GEMM should closely match a plain float Triton GEMM."""
    g = torch.Generator(device="cuda").manual_seed(42)

    a = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)

    assert torch.allclose(
        got, ref, atol=5e-4, rtol=5e-4
    ), f"max error = {(got - ref).abs().max().item()}"


@requires_cuda
def test_gemm_matches_torch():
    """Fixed-point GEMM should closely match torch.matmul."""
    g = torch.Generator(device="cuda").manual_seed(7)

    a = torch.randn((8, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)

    assert torch.allclose(
        got, ref, atol=5e-4, rtol=5e-4
    ), f"max error = {(got - ref).abs().max().item()}"


@requires_cuda
def test_gemm_identity():
    """A @ I = A."""
    K = 16
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((4, K), device="cuda", dtype=torch.float32, generator=g)
    eye = torch.eye(K, device="cuda", dtype=torch.float32)

    got = gemm_fxp_test(a, eye)
    assert torch.allclose(got, a, atol=1e-4)


@requires_cuda
def test_gemm_zero_matrix():
    """A @ 0 = 0."""
    a = torch.ones((4, 16), device="cuda", dtype=torch.float32)
    b = torch.zeros((16, 4), device="cuda", dtype=torch.float32)

    got = gemm_fxp_test(a, b)
    assert torch.equal(got, torch.zeros((4, 4), device="cuda", dtype=torch.float32))


@requires_cuda
def test_gemm_single_row_col():
    """(1,16) @ (16,1) single-element output."""
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((1, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 1), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)
    assert torch.allclose(got, ref, atol=5e-4)


@requires_cuda
def test_gemm_non_square():
    """Non-square dimensions: (2,16) @ (16,3) -> (2,3)."""
    g = torch.Generator(device="cuda").manual_seed(99)
    a = torch.randn((2, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 3), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=5e-4, rtol=5e-4)


@requires_cuda
def test_gemm_associativity_fixed_vs_float16():
    """Demonstrate that fp16 dot-products are order-dependent, while the
    fixed-point GEMM gives identical results for permuted K-dimensions.
    """
    # K=16: first element is large (2048), rest are 1.0.
    # At 2048 the fp16 step size is 2, so 2048+1 = 2048 in fp16.
    # Forward:  (2048 + 1 + 1 + ... + 1) stays at 2048.
    # Reversed: (1 + 1 + ... + 1) = 15, then 15 + 2048 = 2063 → rounds to 2064.
    K = 16
    a_row = [2048.0] + [1.0] * (K - 1)
    b_col = [1.0] * K

    fp16_fwd = _ordered_fp16_dot([a_row], [b_col])
    a_row_rev = list(reversed(a_row))
    b_col_rev = list(reversed(b_col))
    fp16_rev = _ordered_fp16_dot([a_row_rev], [b_col_rev])
    assert (
        fp16_fwd.item() != fp16_rev.item()
    ), "fp16 sums should differ with reversed order"

    a_fwd = torch.tensor([a_row], device="cuda", dtype=torch.float32)
    b_fwd = torch.tensor([[v] for v in b_col], device="cuda", dtype=torch.float32)

    perm = list(reversed(range(K)))
    a_rev = a_fwd[:, perm]
    b_rev = b_fwd[perm, :]

    c_fwd = gemm_fxp_test(a_fwd, b_fwd)
    c_rev = gemm_fxp_test(a_rev, b_rev)

    assert torch.equal(
        c_fwd, c_rev
    ), f"Fixed-point GEMM should be order-invariant, got {c_fwd.item()} vs {c_rev.item()}"


@requires_cuda
def test_gemm_element_permutation_invariance():
    """Arbitrarily permuting the K dimension of both A and B must yield
    bitwise identical results.

    Since every individual product a[m,k]*b[k,n] is converted to
    fixed-point before any reduction, the entire K-sum is an integer
    addition, exact and commutative, so any permutation of K gives
    the same answer.
    """
    g = torch.Generator(device="cuda").manual_seed(123)

    M, K, N = 4, 32, 4
    a = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g)

    perm = torch.randperm(K, device="cuda")
    a_perm = a[:, perm]
    b_perm = b[perm, :]

    c_orig = gemm_fxp_test(a, b)
    c_perm = gemm_fxp_test(a_perm, b_perm)

    assert torch.equal(
        c_orig, c_perm
    ), f"max diff = {(c_orig - c_perm).abs().max().item()}"


@requires_cuda
def test_gemm_deterministic_across_runs():
    """The same inputs must always produce bitwise identical outputs."""
    g = torch.Generator(device="cuda").manual_seed(77)
    a = torch.randn((8, 32), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((32, 8), device="cuda", dtype=torch.float32, generator=g)

    results = [gemm_fxp_test(a, b) for _ in range(5)]
    for r in results[1:]:
        assert torch.equal(results[0], r)
