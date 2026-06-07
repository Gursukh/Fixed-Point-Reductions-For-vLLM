import torch
from .fixed_point_helpers import float_to_fixed, requires_cuda, fixed_to_float


def assert_bitwise_equal_float32(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    # Compare raw bits, not float equality.
    assert torch.equal(a.reshape(1).view(torch.int32), b.reshape(1).view(torch.int32))


def fixed_sum(x: torch.Tensor, out_int: torch.dtype = torch.int32) -> torch.Tensor:
    q = float_to_fixed(x, out_int)
    acc = q.sum()
    info = torch.iinfo(out_int)
    return acc.clamp(info.min, info.max).to(out_int)


def fixed_sum_as_float(
    x: torch.Tensor,
    out_int: torch.dtype = torch.int32,
    out_float: torch.dtype = torch.float32,
) -> torch.Tensor:
    q_sum = fixed_sum(x, out_int).to(torch.int64)
    return fixed_to_float(q_sum, out_float)


def ordered_float_sum(values: list[float], dtype: torch.dtype) -> torch.Tensor:
    acc = torch.zeros((), device="cuda", dtype=dtype)
    for v in values:
        acc = acc + torch.tensor(v, device="cuda", dtype=dtype)
    return acc


def ordered_fixed_sum_as_float(
    values: list[float], out_int: torch.dtype = torch.int64
) -> torch.Tensor:
    x = torch.tensor(values, device="cuda", dtype=torch.float32)
    q = float_to_fixed(x, out_int).to(torch.int64)

    acc = torch.zeros((), device="cuda", dtype=torch.int64)
    for i in range(q.numel()):
        acc = acc + q[i]

    info = torch.iinfo(out_int)
    acc = acc.clamp(info.min, info.max).to(out_int)
    return fixed_to_float(acc.to(torch.int64), torch.float32)


@requires_cuda
def test_sum_on_q_grid_bitwise_identical():
    # Q values picked to round-trip exactly.
    q_vals = torch.tensor(
        [0, 1, -1, 37, -53, 512, -1024, 4095, -777],
        device="cuda",
        dtype=torch.int32,
    )
    x = fixed_to_float(q_vals.to(torch.int64), torch.float32)

    assert torch.equal(float_to_fixed(x, torch.int32), q_vals)

    float_sum = x.sum(dtype=torch.float32)
    fixed_sum_float = fixed_sum_as_float(x)
    assert_bitwise_equal_float32(float_sum, fixed_sum_float)


@requires_cuda
def test_sum_overflow_saturates_to_int32_max():
    # |x| > 32768 saturates each element, so the sum saturates too.
    x = torch.tensor(
        [float(1 << 14), float(1 << 14), 16.0],
        device="cuda",
        dtype=torch.float32,
    )
    q_sum = fixed_sum(x, out_int=torch.int32)
    assert q_sum.item() == torch.iinfo(torch.int32).max

    got = fixed_to_float(q_sum.to(torch.int64), torch.float32)
    expected = fixed_to_float(
        torch.tensor(torch.iinfo(torch.int32).max, device="cuda", dtype=torch.int64),
        torch.float32,
    )
    assert_bitwise_equal_float32(got, expected)


@requires_cuda
def test_sum_underflow_saturates_to_int32_min():
    x = torch.tensor(
        [-float(1 << 14), -float(1 << 14), -16.0],
        device="cuda",
        dtype=torch.float32,
    )
    q_sum = fixed_sum(x, out_int=torch.int32)
    assert q_sum.item() == torch.iinfo(torch.int32).min

    got = fixed_to_float(q_sum.to(torch.int64), torch.float32)
    expected = fixed_to_float(
        torch.tensor(torch.iinfo(torch.int32).min, device="cuda", dtype=torch.int64),
        torch.float32,
    )
    assert_bitwise_equal_float32(got, expected)


@requires_cuda
def test_associativity_float_non_assoc_fixed_assoc():
    # fp16 loses the +1 at large magnitudes; int64 fxp keeps it.
    order_a = [65504.0, -65504.0, 1.0]
    order_b = [65504.0, 1.0, -65504.0]

    float_a = ordered_float_sum(order_a, dtype=torch.float16)
    float_b = ordered_float_sum(order_b, dtype=torch.float16)
    assert float_a.item() != float_b.item()

    fixed_a = ordered_fixed_sum_as_float(order_a, out_int=torch.int64)
    fixed_b = ordered_fixed_sum_as_float(order_b, out_int=torch.int64)
    assert_bitwise_equal_float32(fixed_a, fixed_b)

    expected = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    assert_bitwise_equal_float32(fixed_a, expected)
