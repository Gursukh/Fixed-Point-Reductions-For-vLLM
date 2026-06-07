import pytest
import torch

from .fixed_point_helpers import (
    gemm_fxp_test,
    requires_cuda,
    skip_if_dtype_unsupported,
)


@requires_cuda
@pytest.mark.parametrize("M,N,K", [(4, 4, 16), (3, 7, 16), (1, 1, 32), (8, 16, 32)])
def test_gemm_correctness(M, N, K):
    skip_if_dtype_unsupported(torch.float32)
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
    skip_if_dtype_unsupported(torch.float32)
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
    skip_if_dtype_unsupported(torch.float32)
    K = 16
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((4, K), device="cuda", dtype=torch.float32, generator=g)
    eye = torch.eye(K, device="cuda", dtype=torch.float32)

    got = gemm_fxp_test(a, eye)
    assert torch.allclose(got, a, atol=1e-4)


@requires_cuda
def test_gemm_zero_matrix():
    skip_if_dtype_unsupported(torch.float32)
    a = torch.ones((4, 16), device="cuda", dtype=torch.float32)
    b = torch.zeros((16, 4), device="cuda", dtype=torch.float32)

    got = gemm_fxp_test(a, b)
    assert torch.equal(got, torch.zeros((4, 4), device="cuda", dtype=torch.float32))


@requires_cuda
def test_gemm_single_row_col():
    skip_if_dtype_unsupported(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn((1, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 1), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)
    assert torch.allclose(got, ref, atol=5e-4)


@requires_cuda
def test_gemm_non_square():
    skip_if_dtype_unsupported(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(99)
    a = torch.randn((2, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 3), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b)
    ref = torch.matmul(a, b)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=5e-4, rtol=5e-4)


_DTYPE_TOL = {
    torch.float32: (5e-4, 5e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (5e-2, 5e-2),
}


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [(8, 16, 32), (4, 4, 64)])
def test_gemm_native_dtype_vec_path(dtype, M, N, K):
    skip_if_dtype_unsupported(dtype)
    g = torch.Generator(device="cuda").manual_seed(42)
    a_f32 = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g) * 0.5
    b_f32 = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g) * 0.5

    a = a_f32.to(dtype)
    b = b_f32.to(dtype)

    got = gemm_fxp_test(a, b)
    assert got.dtype == dtype, f"output dtype {got.dtype} != input dtype {dtype}"

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(dtype)

    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(
        got.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol
    ), f"max error = {(got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()} dtype={dtype}"


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm_native_dtype_scalar_fallback(dtype):
    skip_if_dtype_unsupported(dtype)
    # K=33 forces the scalar K-tail path.
    M, N, K = 4, 4, 33
    g = torch.Generator(device="cuda").manual_seed(0)
    a_f32 = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g) * 0.5
    b_f32 = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g) * 0.5

    a = a_f32.to(dtype)
    b = b_f32.to(dtype)

    got = gemm_fxp_test(a, b)
    assert got.dtype == dtype

    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(dtype)
    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(
        got.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol
    )


@requires_cuda
def test_gemm_deterministic_across_runs():
    skip_if_dtype_unsupported(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(77)
    a = torch.randn((8, 32), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((32, 8), device="cuda", dtype=torch.float32, generator=g)

    results = [gemm_fxp_test(a, b) for _ in range(5)]
    for r in results[1:]:
        assert torch.equal(results[0], r)


@requires_cuda
def test_gemm_bias_none_matches_no_bias():
    skip_if_dtype_unsupported(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(11)
    a = torch.randn((8, 16), device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32, generator=g)

    with_none = gemm_fxp_test(a, b, bias=None)
    ref = torch.matmul(a, b)
    assert torch.allclose(with_none, ref, atol=5e-4, rtol=5e-4)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [(8, 16, 32), (4, 4, 64), (130, 130, 33)])
def test_gemm_with_bias(dtype, M, N, K):
    skip_if_dtype_unsupported(dtype)
    g = torch.Generator(device="cuda").manual_seed(123)
    a_f32 = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g) * 0.5
    b_f32 = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g) * 0.5
    bias_f32 = torch.randn((N,), device="cuda", dtype=torch.float32, generator=g)

    a = a_f32.to(dtype)
    b = b_f32.to(dtype)
    bias = bias_f32.to(dtype)

    got = gemm_fxp_test(a, b, bias=bias)
    assert got.dtype == dtype
    assert got.shape == (M, N)

    ref = (
        torch.matmul(a.to(torch.float32), b.to(torch.float32)) + bias_f32
    ).to(dtype)

    atol, rtol = _DTYPE_TOL[dtype]
    assert torch.allclose(
        got.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol
    ), f"max error = {(got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()} dtype={dtype}"


@requires_cuda
def test_gemm_large_shape():
    skip_if_dtype_unsupported(torch.float32)
    # Big enough to span several 128x128 tiles in both dims.
    M, K, N = 256, 256, 256
    g = torch.Generator(device="cuda").manual_seed(5)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32, generator=g) * 0.3
    b = torch.randn((K, N), device="cuda", dtype=torch.float32, generator=g) * 0.3
    bias = torch.randn((N,), device="cuda", dtype=torch.float32, generator=g)

    got = gemm_fxp_test(a, b, bias=bias)
    ref = torch.matmul(a, b) + bias

    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3), (
        f"max error = {(got - ref).abs().max().item()}"
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm_ktile_permutation_invariance(dtype):
    # Permuting whole K-tiles is bit-exact. Per-element permutation is not.
    skip_if_dtype_unsupported(dtype)
    g = torch.Generator(device="cuda").manual_seed(123)

    # Pull BLOCK_K from the kernel so this stays correct if it gets retuned.
    from fxpr_vllm._triton.gemm import _BLOCK_K_BY_DTYPE
    K_TILE = _BLOCK_K_BY_DTYPE[dtype]
    NUM_TILES = 4
    M, K, N = 4, K_TILE * NUM_TILES, 4
    a = torch.randn((M, K), device="cuda", dtype=dtype, generator=g)
    b = torch.randn((K, N), device="cuda", dtype=dtype, generator=g)

    tile_perm = torch.randperm(NUM_TILES, device="cuda")
    perm = torch.cat([
        torch.arange(t * K_TILE, (t + 1) * K_TILE, device="cuda")
        for t in tile_perm
    ])
    a_perm = a[:, perm]
    b_perm = b[perm, :]

    c_orig = gemm_fxp_test(a, b)
    c_perm = gemm_fxp_test(a_perm, b_perm)

    assert torch.equal(
        c_orig, c_perm
    ), f"max diff = {(c_orig - c_perm).abs().max().item()}"


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm_batch_size_invariance(dtype):
    # The whole point of fxp: c[m, :] is identical no matter the batch size.
    skip_if_dtype_unsupported(dtype)
    g = torch.Generator(device="cuda").manual_seed(2026)

    K, N = 64, 16
    a_full = torch.randn((128, K), device="cuda", dtype=dtype, generator=g)
    b = torch.randn((K, N), device="cuda", dtype=dtype, generator=g)

    rows = [0, 3, 17, 64, 127]
    refs = {m: gemm_fxp_test(a_full[m:m + 1], b)[0] for m in rows}

    for batch in (1, 4, 16, 128):
        for m in rows:
            if m >= batch:
                continue
            sub = a_full[:batch].clone()
            sub[m] = a_full[m]
            out = gemm_fxp_test(sub, b)
            assert torch.equal(out[m], refs[m]), (
                f"row {m} differs at batch={batch}, dtype={dtype}: "
                f"max diff = {(out[m] - refs[m]).abs().max().item()}"
            )


@requires_cuda
@pytest.mark.parametrize("int_bits", [32, 64])
def test_gemm_parametrized_int_bits(int_bits):
    skip_if_dtype_unsupported(torch.float32)
    g = torch.Generator(device="cuda").manual_seed(int_bits)

    # Keep values small so partial products don't overflow int_bits=32.
    a = torch.randn((4, 16), device="cuda", dtype=torch.float32, generator=g) * 0.25
    b = torch.randn((16, 4), device="cuda", dtype=torch.float32, generator=g) * 0.25

    got = gemm_fxp_test(a, b, fxp_int_bits=int_bits)
    ref = torch.matmul(a, b)

    assert torch.allclose(got, ref, atol=5e-2, rtol=5e-2), (
        f"max error = {(got - ref).abs().max().item()} at int_bits={int_bits}"
    )
