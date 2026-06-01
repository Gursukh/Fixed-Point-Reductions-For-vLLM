"""Float <-> signed fixed-point helpers shared by every kernel.

float_to_fixed scales, rounds half-to-even, clamps and casts in one PTX
cvt.rni.sat for int16/int32. int64 has no .sat, so it clamps in float first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_INT_DTYPE_BY_BITS = {
    16: tl.int16,
    32: tl.int32,
    64: tl.int64,
}

_TORCH_INT_DTYPE_BY_BITS = {
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}

# Memoise (int_bits, frac_bits) -> derived constants; launchers call this once
# per layer per step, so keep it off the CPU hot path.
_CONSTANTS_CACHE: dict[tuple[int, int], tuple] = {}


def fxp_constants(int_bits: int, frac_bits: int):
    key = (int_bits, frac_bits)
    cached = _CONSTANTS_CACHE.get(key)
    if cached is not None:
        return cached
    if int_bits not in _INT_DTYPE_BY_BITS:
        raise ValueError(f"int_bits must be 16/32/64, got {int_bits}")
    if not (0 <= frac_bits < 64):
        raise ValueError(f"frac_bits must satisfy 0 <= N < 64, got {frac_bits}")
    scale = float(1 << frac_bits)
    inv_scale = 1.0 / scale
    qmax_i = (1 << (int_bits - 1)) - 1
    qmin_i = -(1 << (int_bits - 1))
    result = (
        scale,
        inv_scale,
        float(qmin_i),
        float(qmax_i),
        _INT_DTYPE_BY_BITS[int_bits],
        _TORCH_INT_DTYPE_BY_BITS[int_bits],
    )
    _CONSTANTS_CACHE[key] = result
    return result


@triton.jit
def _cvt_rni_sat_s32_f32(x):
    return tl.inline_asm_elementwise(
        "cvt.rni.sat.s32.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_rni_sat_s16_f32(x):
    return tl.inline_asm_elementwise(
        "cvt.rni.sat.s16.f32 $0, $1;",
        "=h,r",
        [x],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_rni_s64_f32(x):
    return tl.inline_asm_elementwise(
        "cvt.rni.s64.f32 $0, $1;",
        "=l,r",
        [x],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )


@triton.jit
def float_to_fixed(
    x,
    SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
):
    scaled = x.to(tl.float32) * SCALE
    if INT_DTYPE == tl.int32:
        return _cvt_rni_sat_s32_f32(scaled)
    elif INT_DTYPE == tl.int16:
        return _cvt_rni_sat_s16_f32(scaled)
    else:
        # int64: no .sat for s64.f32, clamp in float first.
        clamped = tl.minimum(tl.maximum(scaled, QMIN), QMAX)
        return _cvt_rni_s64_f32(clamped)


@triton.jit
def fixed_to_float(x, INV_SCALE: tl.constexpr):
    return x.to(tl.float32) * INV_SCALE
