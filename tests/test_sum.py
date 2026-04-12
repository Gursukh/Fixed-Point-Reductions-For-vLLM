import pytest
import torch
from tests.fixed_point_helpers import flp2fxp, requires_cuda, fxp2flp


def assert_bitwise_equal_float32(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    # Reinterpret as int32 so equality means exact bitwise match.
    assert torch.equal(a.reshape(1).view(torch.int32), b.reshape(1).view(torch.int32))


def fixed_sum(
    x: torch.Tensor, exponent: int, out_int: torch.dtype = torch.int32
) -> torch.Tensor:
    q = flp2fxp(x, exponent, out_int)
    acc = q.sum()
    info = torch.iinfo(out_int)
    return acc.clamp(info.min, info.max).to(out_int)


def fixed_sum_as_float(
    x: torch.Tensor,
    exponent: int,
    out_int: torch.dtype = torch.int32,
    out_float: torch.dtype = torch.float32,
) -> torch.Tensor:
    q_sum = fixed_sum(x, exponent, out_int).to(torch.int64)
    return fxp2flp(q_sum, exponent, out_float)


def ordered_float_sum(values: list[float], dtype: torch.dtype) -> torch.Tensor:
    acc = torch.zeros((), device="cuda", dtype=dtype)
    for v in values:
        acc = acc + torch.tensor(v, device="cuda", dtype=dtype)
    return acc


def ordered_fixed_sum_as_float(values: list[float], exponent: int) -> torch.Tensor:
    x = torch.tensor(values, device="cuda", dtype=torch.float32)
    q = flp2fxp(x, exponent, torch.int32).to(torch.int64)

    acc = torch.zeros((), device="cuda", dtype=torch.int64)
    for i in range(q.numel()):
        acc = acc + q[i]

    info = torch.iinfo(torch.int32)
    acc = acc.clamp(info.min, info.max).to(torch.int32)
    return fxp2flp(acc.to(torch.int64), exponent, torch.float32)


@requires_cuda
@pytest.mark.parametrize("exponent", [4, 8, 16])
def test_sum_on_q_grid_bitwise_identical(exponent):
    # Build values directly from fixed integers so every float lands on the Q grid.
    q_vals = torch.tensor(
        [0, 1, -1, 37, -53, 512, -1024, 4095, -777],
        device="cuda",
        dtype=torch.int32,
    )
    x = fxp2flp(q_vals.to(torch.int64), exponent, torch.float32)

    # Sanity: values are exactly on-grid for this exponent.
    assert torch.equal(flp2fxp(x, exponent, torch.int32), q_vals)

    float_sum = x.sum(dtype=torch.float32)
    fixed_sum_float = fixed_sum_as_float(
        x, exponent, out_int=torch.int32, out_float=torch.float32
    )
    assert_bitwise_equal_float32(float_sum, fixed_sum_float)


@requires_cuda
def test_sum_overflow_saturates_to_int32_max():
    exponent = 0
    x = torch.tensor(
        [float(1 << 30), float(1 << 30), 16.0],
        device="cuda",
        dtype=torch.float32,
    )
    q_sum = fixed_sum(x, exponent, out_int=torch.int32)
    assert q_sum.item() == torch.iinfo(torch.int32).max

    got = fxp2flp(q_sum.to(torch.int64), exponent, torch.float32)
    expected = fxp2flp(
        torch.tensor(torch.iinfo(torch.int32).max, device="cuda", dtype=torch.int64),
        exponent,
        torch.float32,
    )
    assert_bitwise_equal_float32(got, expected)


@requires_cuda
def test_sum_underflow_saturates_to_int32_min():
    exponent = 0
    x = torch.tensor(
        [-float(1 << 30), -float(1 << 30), -16.0],
        device="cuda",
        dtype=torch.float32,
    )
    q_sum = fixed_sum(x, exponent, out_int=torch.int32)
    assert q_sum.item() == torch.iinfo(torch.int32).min

    got = fxp2flp(q_sum.to(torch.int64), exponent, torch.float32)
    expected = fxp2flp(
        torch.tensor(torch.iinfo(torch.int32).min, device="cuda", dtype=torch.int64),
        exponent,
        torch.float32,
    )
    assert_bitwise_equal_float32(got, expected)


@requires_cuda
def test_associativity_float_non_assoc_fixed_assoc():
    # In fp16, one order loses the +1 due to rounding at large magnitude.
    order_a = [65504.0, -65504.0, 1.0]
    order_b = [65504.0, 1.0, -65504.0]

    float_a = ordered_float_sum(order_a, dtype=torch.float16)
    float_b = ordered_float_sum(order_b, dtype=torch.float16)
    assert float_a.item() != float_b.item()

    fixed_a = ordered_fixed_sum_as_float(order_a, exponent=0)
    fixed_b = ordered_fixed_sum_as_float(order_b, exponent=0)
    assert_bitwise_equal_float32(fixed_a, fixed_b)

    expected = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    assert_bitwise_equal_float32(fixed_a, expected)
