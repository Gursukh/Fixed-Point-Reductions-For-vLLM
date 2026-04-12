import triton
import triton.language as tl


@triton.jit
def flp_2_fxp(
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
def fxp_to_flp(
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
