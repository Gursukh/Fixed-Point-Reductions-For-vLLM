import pytest
import torch
from tests.fixed_point_helpers import float_to_fixed, requires_cuda, fixed_to_float


@requires_cuda
def test_float_to_fixed_known_values():
    # Q16.16: fixed = round(x * 2^16)
    frac_bits = 16

    # All values are be exactly representable in the fixed-point format,
    # so no rounding error is expected.
    x = torch.tensor(
        [
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.25,
            1.5,
            -1.5,
            2**15 - 1,
            -(2**15),
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            0,
            1 << 16,
            -(1 << 16),
            1 << 15,
            -(1 << 15),
            1 << 14,
            (1 << 16) + (1 << 15),
            -((1 << 16) + (1 << 15)),
            (2**15 - 1) << 16,
            -(2**15) << 16,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    got = float_to_fixed(x, frac_bits, torch.int32)
    assert torch.equal(got, expected)


@requires_cuda
def test_float_to_fixed_saturation():
    x = torch.tensor([1e30, -1e30, 0.0], device="cuda", dtype=torch.float32)
    got = float_to_fixed(x, 0, torch.int32)
    imax = torch.iinfo(torch.int32).max
    imin = torch.iinfo(torch.int32).min

    assert imax - 256 <= got[0].item() <= imax
    assert imin <= got[1].item() <= imin + 256
    assert got[2].item() == 0


@requires_cuda
@pytest.mark.parametrize("frac_bits", [0, 4, 16])
def test_float_roundtrip_lossless(frac_bits):
    step = 2**-frac_bits
    candidates = torch.tensor(
        [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 0.125, 2.0, -2.0, 10.0, -10.0],
        device="cuda",
        dtype=torch.float32,
    )
    # keep only values that land exactly on the Q-format grid
    base = candidates[(candidates / step).frac() == 0]
    q = float_to_fixed(base, frac_bits, torch.int32)
    back = fixed_to_float(q, frac_bits, torch.float32)
    assert torch.equal(back, base), f"{base} -> {q} -> {back}"


@requires_cuda
def test_float_roundtrip_rounding():
    frac_bits = 4
    step = 2**-frac_bits
    # pick values strictly between grid points
    x = torch.tensor(
        [0.03, 0.1, 0.2, -0.03, -0.1, 0.7777], device="cuda", dtype=torch.float32
    )
    back = fixed_to_float(
        float_to_fixed(x, frac_bits, torch.int32), frac_bits, torch.float32
    )

    # result must land on the grid
    grid_err = (back / step).round() * step - back
    assert torch.allclose(grid_err, torch.zeros_like(grid_err), atol=1e-7)

    # and must be within half a step of the original (round-to-nearest)
    assert torch.all((x - back).abs() <= step / 2 + 1e-7)

    # and must actually differ (otherwise the test picked lossless values)
    assert torch.any(x != back)


@requires_cuda
@pytest.mark.parametrize("frac_bits", [0, 8, 16])
def test_fixed_roundtrip_lossless(frac_bits):
    # int32 values small enough that float32 (24-bit mantissa) represents them exactly
    q = torch.tensor(
        [0, 1, -1, 123, -123, (1 << 20), -(1 << 20)], device="cuda", dtype=torch.int32
    )
    f = fixed_to_float(q, frac_bits, torch.float32)
    back = float_to_fixed(f, frac_bits, torch.int32)
    assert torch.equal(back, q)


@requires_cuda
def test_fixed_roundtrip_rounding():
    # float32 has 24 bits of mantissa, so int32 values above 2^24 can't be represented exactly
    frac_bits = 0
    q = torch.tensor(
        [(1 << 24) + 1, (1 << 25) + 3, -((1 << 25) + 3)],
        device="cuda",
        dtype=torch.int32,
    )
    f = fixed_to_float(q, frac_bits, torch.float32)
    back = float_to_fixed(f, frac_bits, torch.int32)

    # something must have changed
    assert not torch.equal(back, q)
    # but the error is bounded by the float32 ulp at that magnitude
    ulp = torch.tensor([2.0, 4.0, 4.0], device="cuda", dtype=torch.float32)
    assert torch.all((back - q).abs().to(torch.float32) <= ulp)
