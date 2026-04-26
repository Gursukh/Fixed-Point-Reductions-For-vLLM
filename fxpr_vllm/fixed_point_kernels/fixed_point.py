import triton  # noqa: F401  (kept for downstream re-exports)
import triton.language as tl

RCP_LN2 = 1.4426950408889634

_TL_INT_BY_BITS = {16: tl.int16, 32: tl.int32, 64: tl.int64}


def fixed_tl_dtype(int_bits: int):
    """Return the tl signed-int dtype for a given bit width (16/32/64)."""
    try:
        return _TL_INT_BY_BITS[int_bits]
    except KeyError:
        raise ValueError(f"fxp_int_bits must be 16/32/64, got {int_bits}") from None


def int_bits_of(fxp_dtype) -> int:
    """Inverse of :func:`fixed_tl_dtype`: bit width of a tl signed-int dtype."""
    bits = getattr(fxp_dtype, "primitive_bitwidth", None)
    if bits not in (16, 32, 64):
        raise ValueError(f"Unsupported fxp dtype {fxp_dtype!r}")
    return bits


@triton.jit
def float_to_fixed(
    x: tl.tensor, fractional_bit_width: tl.constexpr, fixed_point_type: tl.constexpr
):

    tl.static_assert(
        fixed_point_type == tl.int16
        or fixed_point_type == tl.int32
        or fixed_point_type == tl.int64,
        "Fixed-point conversion must use a signed integer dtype",
    )
    tl.static_assert(
        x.dtype == tl.float16 or x.dtype == tl.float32 or x.dtype == tl.float64,
        "x must be of a floating-point type",
    )

    bits: tl.constexpr = fixed_point_type.primitive_bitwidth
    mantissa_bits: tl.constexpr = x.dtype.fp_mantissa_width + 1

    safe_shift: tl.constexpr = 0 if bits <= mantissa_bits else (bits - mantissa_bits)
    qmax_f: tl.constexpr = float((1 << (bits - 1)) - (1 << safe_shift))
    qmin_f: tl.constexpr = float(-(1 << (bits - 1)))

    scale = 2.0**fractional_bit_width
    scaled = x * scale
    clamped = tl.minimum(tl.maximum(scaled, qmin_f), qmax_f)

    rounded = tl.extra.libdevice.rint(clamped)
    return rounded.to(fixed_point_type)


@triton.jit
def fixed_to_float(
    x: tl.tensor, fractional_bit_width: tl.constexpr, floating_point_type: tl.constexpr
):

    tl.static_assert(
        floating_point_type == tl.float16
        or floating_point_type == tl.float32
        or floating_point_type == tl.float64,
        "Fixed-point conversion must use a float dtype",
    )

    inv_scale = 2.0**-fractional_bit_width
    return x.to(floating_point_type) * inv_scale
