"""Tensor-core fixed-point GEMM. Persistent kernel for prefill, split-K kernel
for tall-skinny decode. Both quantise each per-K-tile fp32 partial to a
fixed-point int and accumulate in ints, so the result is bit-identical no matter
how the problem is tiled or split.
"""

from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl

from .fxp import fxp_constants, float_to_fixed, fixed_to_float


# BLOCK_K is the determinism granularity: each kernel quantises one fp32 tl.dot
# partial per BLOCK_K chunk, and both kernels must share it so their int sums
# bit-match (a token routes through either kernel depending on batch size).
# Bumped from 32/16 to 128/64 so fewer, larger tl.dot calls amortise the
# cvt.rni.sat + add, roughly 4x fewer quantise sweeps. Safe for int32/frac16:
# float_to_fixed only saturates above abs(partial) > 32768 and a BLOCK_K-wide
# partial is O(1..100). fp32 stays 64 - its tiles are 2x the bytes so 128 won't
# fit shared memory.
_BLOCK_K_BY_DTYPE = {
    torch.float16: 128,
    torch.bfloat16: 128,
    torch.float32: 64,
}
_SUPPORTED_DTYPES = tuple(_BLOCK_K_BY_DTYPE.keys())

_TORCH_TO_TL_FLOAT = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# Fixed tiling the split-K kernel always runs with:
# (BLOCK_M, BLOCK_N, GROUP_SIZE_M, num_warps, num_stages). Only BLOCK_K affects
# numerics, so BM/BN/num_stages are free to tune (BM, BN >= 16 keeps tl.dot on
# tensor cores). Smaller tiles and fewer warps than the persistent configs to cut
# register pressure - the old (128, 64, 8 warps) hit 172 reg/thread and 1
# block/SM (16.6% occupancy) on Blackwell. Atomic-add latency hiding wants higher
# occupancy; 4 stages let the compiler pipeline the cp.async loads.
_CFG_SPLITK = (64, 64, 8, 4, 4)

_BLOCKS_PER_SM = 2

# Persistent split-K scratch, reused across calls. c_int is at most
# target_programs * 128*128 ints; the epilogue self-zeroes it.
_TILE_M_MAX = 128
_TILE_N_MAX = 128

# The split-K kernel needs one int32 lock per (m,n) tile, indexed by tile_id_mn
# in [0, sk_num_tiles_mn). A 128*128 c_int slab holds this many 64x64 split-K
# tiles, so target_programs slabs hold target_programs times that. An N not on a
# 64 boundary can add one extra tile per row-tile, and autotune never probes past
# 16 row-tiles, so a +16 margin covers any shape that fits c_int.
_MAX_TILES_PER_PROGRAM = (_TILE_M_MAX // _CFG_SPLITK[0]) * (_TILE_N_MAX // _CFG_SPLITK[1])
_SPLITK_LOCK_MARGIN = 16


def _splitk_locks_len(target_programs: int) -> int:
    """Lock-buffer length, big enough for the most tiles any clamp-passing
    split-K shape can produce. _resolve_split_k refuses to split past it."""
    return target_programs * _MAX_TILES_PER_PROGRAM + _SPLITK_LOCK_MARGIN


_splitk_scratch: dict[
    tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def _get_splitk_scratch(
    device: torch.device, int_dtype: torch.dtype, target_programs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent (c_int_flat, locks_flat) scratch for the split-K path. Both
    sized for the worst case (c_int by ints, locks by tile count) and zeroed once;
    the epilogue rezeroes its own footprint. Allocated during warmup before graph
    capture and fixed in size, so it never moves."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, int_dtype)
    cached = _splitk_scratch.get(key)
    if cached is None:
        c_int = torch.zeros(
            target_programs * _TILE_M_MAX * _TILE_N_MAX,
            device=device,
            dtype=int_dtype,
        )
        locks = torch.zeros(
            _splitk_locks_len(target_programs), device=device, dtype=torch.int32
        )
        cached = (c_int, locks)
        _splitk_scratch[key] = cached
    return cached


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# do_not_specialize M: it changes per request and recompiling per M class would
# spike TTFT; M is only a bound/mask anyway. M_bucket is just an autotune cache
# key, never read in the body, so don't specialize on it either.
@triton.jit(
    do_not_specialize=["M", "M_bucket"],
    do_not_specialize_on_alignment=["M", "M_bucket"],
)
def _gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    M_bucket,  # autotune key only, ignored in body
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,  # launched grid size = persistent loop stride
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_m_raw = start_m + tl.arange(0, BLOCK_M)
        offs_n_raw = start_n + tl.arange(0, BLOCK_N)

        offs_am = tl.where(offs_m_raw < M, offs_m_raw, 0)
        offs_bn = tl.where(offs_n_raw < N, offs_n_raw, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        # Build pointers once, advance by BLOCK_K per K iter so the inner loop is
        # an add instead of re-multiplying offsets.
        offs_k0 = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k0[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k0[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(k_tiles):
            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_fp32 = fixed_to_float(fxp_acc, INV_SCALE)
        if HAS_BIAS:
            bias_vals = tl.load(
                bias_ptr + offs_n_raw, mask=offs_n_raw < N, other=0.0
            ).to(tl.float32)
            c_fp32 = c_fp32 + bias_vals[None, :]

        c_ptrs = c_ptr + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.store(c_ptrs, c_fp32.to(IO_DTYPE), mask=c_mask)


# do_not_specialize the shape-varying args so this compiles once. SPLIT_K is a
# runtime arg, not constexpr: it depends on M, so it can't be enumerated.
@triton.jit(
    do_not_specialize=["M", "SPLIT_K"],
    do_not_specialize_on_alignment=["M", "SPLIT_K"],
)
def _gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_int_ptr,
    c_ptr,
    lock_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cim,
    stride_cin,
    stride_cm,
    stride_cn,
    num_k_tiles,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT_DTYPE: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles_mn = num_pid_m * num_pid_n
    num_tiles = num_tiles_mn * SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_k = tile_id // num_tiles_mn
        tile_id_mn = tile_id % num_tiles_mn
        pid_m, pid_n = _compute_pid(
            tile_id_mn, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_m_raw = start_m + tl.arange(0, BLOCK_M)
        offs_n_raw = start_n + tl.arange(0, BLOCK_N)

        offs_am = tl.where(offs_m_raw < M, offs_m_raw, 0)
        offs_bn = tl.where(offs_n_raw < N, offs_n_raw, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        # Interleaved K stride: pid_k=0 takes tiles 0, SPLIT_K, 2*SPLIT_K, ...,
        # pid_k=1 takes 1, SPLIT_K+1, ..., so all SPLIT_K programs touch adjacent
        # A/B K-tiles each iter for better L2 reuse. Integer atomic_add is
        # commutative, so this is bit-identical to contiguous chunks. Build
        # pointers once, advance by BLOCK_K*SPLIT_K per iter.
        offs_k0 = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k0[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k0[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        adv_ak = BLOCK_K * SPLIT_K * stride_ak
        adv_bk = BLOCK_K * SPLIT_K * stride_bk

        fxp_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE)
        for ki in range(pid_k, num_k_tiles, SPLIT_K):
            k_mask = offs_k_for_mask < K - ki * BLOCK_K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            partial = tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=tl.float32)
            fxp_acc += float_to_fixed(partial, SCALE, QMIN, QMAX, INT_DTYPE)

            a_ptrs += adv_ak
            b_ptrs += adv_bk

        # Integer atomic-add is commutative, so this is race-free.
        c_int_ptrs = (
            c_int_ptr + offs_m_raw[:, None] * stride_cim + offs_n_raw[None, :] * stride_cin
        )
        c_mask = (offs_m_raw[:, None] < M) & (offs_n_raw[None, :] < N)
        tl.atomic_add(c_int_ptrs, fxp_acc, mask=c_mask, sem="relaxed")

        # Fused epilogue, owned by the last SPLIT_K program to arrive. The acq_rel
        # increment makes its load of c_int see every split's atomic-add, so the
        # output is order-independent.
        arrived = tl.atomic_add(lock_ptr + tile_id_mn, 1, sem="acq_rel")
        if arrived == SPLIT_K - 1:
            fxp = tl.load(c_int_ptrs, mask=c_mask, other=0)
            c_fp32 = fixed_to_float(fxp, INV_SCALE)
            if HAS_BIAS:
                bias_vals = tl.load(
                    bias_ptr + offs_n_raw, mask=offs_n_raw < N, other=0.0
                ).to(tl.float32)
                c_fp32 = c_fp32 + bias_vals[None, :]
            c_ptrs = (
                c_ptr + offs_m_raw[:, None] * stride_cm + offs_n_raw[None, :] * stride_cn
            )
            tl.store(c_ptrs, c_fp32.to(IO_DTYPE), mask=c_mask)

            # Zero this tile's scratch for the next launch. All splits landed
            # and nothing else touches it, so this is race-free.
            tl.store(
                c_int_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=INT_DTYPE), mask=c_mask
            )
            tl.store(lock_ptr + tile_id_mn, 0)


def _cap() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# Per-dtype arch check. A dtype enters the cache after its first OK call, so
# later calls are just a set-membership test.
_ARCH_CHECKED_DTYPES: set[torch.dtype] = set()


def _check_arch(dtype: torch.dtype) -> None:
    if dtype in _ARCH_CHECKED_DTYPES:
        return
    cap = _cap()
    if cap < 75:
        raise RuntimeError(
            f"gemm_fxp: requires compute capability >= 7.5 (Turing); device is "
            f"{cap // 10}.{cap % 10}"
        )
    if dtype in (torch.bfloat16, torch.float32) and cap < 80:
        name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        raise RuntimeError(
            f"gemm_fxp: {name} inputs require compute capability >= 8.0 "
            f"(Ampere); device is {cap // 10}.{cap % 10}. Use float16 inputs."
        )
    _ARCH_CHECKED_DTYPES.add(dtype)


# Cache SM count - gemm_fxp_run runs for every linear layer.
_SM_COUNT_CACHE: dict[int, int] = {}


def _device_sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _SM_COUNT_CACHE.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _SM_COUNT_CACHE[idx] = n
    return n


# Splitting a GEMM's K reduction helps some shapes and hurts others, and no
# formula predicts which - the crossover even flips between A100 and Blackwell for
# the same shape. Both paths give identical bits, so it's purely a speed question.
# Warmup times both on the real GPU and caches the winner here
# (autotune_split_k); anything warmup didn't probe stays persistent.
_SPLITK_SPLIT = 2          # split count autotune tries (and the cap)

# Measured split choice (1 or 2), keyed by (sk_pid_m, N, K, dtype, int_bits).
# sk_pid_m is the split-K kernel's row-tile count, ceil(M / BLOCK_M). Keying on
# it rather than the coarser autotune bucket keeps batch sizes that want
# different paths from sharing an entry. Missing key means stay persistent.
_SPLITK_CHOICE: dict[tuple, int] = {}
# Set only while autotune_split_k is timing, to pin a path. None otherwise.
_SPLITK_FORCE: int | None = None


def _resolve_split_k(
    sk_pid_m: int,
    num_k_tiles: int,
    int_bits: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    scratch_ints: int,
    sk_num_tiles_mn: int,
    locks_capacity: int,
) -> int:
    """Pick the split count for this call. While warmup is timing, use the pinned
    value; otherwise use what warmup measured, defaulting to persistent. Clamped
    so it can't exceed the K-tiles, overrun the c_int scratch (M*N), or index past
    the lock buffer (one lock per (m,n) tile)."""
    if int_bits == 16:   # int16 atomics aren't widely supported
        return 1
    if _SPLITK_FORCE is not None:
        split = _SPLITK_FORCE
    else:
        split = _SPLITK_CHOICE.get((sk_pid_m, N, K, dtype, int_bits), 1)
    if split <= 1:
        return 1
    split = min(split, num_k_tiles)
    if split < 2 or M * N > scratch_ints or sk_num_tiles_mn > locks_capacity:
        return 1
    return split


# Coarse pow2-ish buckets bound the autotune cache. Without them the tuner would
# re-evaluate per request-size M, spiking TTFT on misses. Spans decode (1..256)
# and prefill (up to a few k tokens).
_M_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _bucket_m(M: int) -> int:
    for b in _M_BUCKETS:
        if M <= b:
            return b
    return _M_BUCKETS[-1]


# Candidate Configs (BM, BN, GROUP_SIZE_M, num_warps, num_stages) span every
# server arch from L4 (sm_89, ~100 KB SMEM, 30 SMs) up to A100/H100/Blackwell
# (up to ~228 KB SMEM, 100+ SMs). Configs that don't fit a GPU's SMEM budget fail
# compilation and triton.autotune silently drops them, so it's safe to list
# large-tile entries that only Hopper/Blackwell can run.
#
# Determinism isn't in the search space: BLOCK_K is pinned per dtype by the caller
# and never enters a Config, so every Config gives the same fxp sum.
#
# SMEM per stage for bf16/fp16 (BK=128, 2 B/elem): 2 * (BM*BK + BK*BN) * 2 B.
# Reference points (fits L4 if total <= ~96 KB):
#   (BM= 16, BN= 64): per_stage=20 KB; ns=2 -> 40 KB,  ns=4 ->  80 KB
#   (BM= 16, BN=128): per_stage=36 KB; ns=2 -> 72 KB,  ns=3 -> 108 KB (A100+)
#   (BM= 64, BN=128): per_stage=48 KB; ns=2 -> 96 KB,  ns=3 -> 144 KB (A100+)
#   (BM=128, BN=128): per_stage=64 KB; ns=2 ->128 KB (A100+), ns=3 ->192 KB (H100)
#   (BM=256, BN=128): per_stage=96 KB; ns=2 ->192 KB (H100+)
_AUTOTUNE_CONFIGS = [
    # ---- Skinny-M (decode, lm_head): tiny BM ----
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=2, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # Hopper/Blackwell - wider N, fewer programs
        {"BLOCK_M": 16, "BLOCK_N": 256, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),

    # ---- Mid-M (small-batch decode, post-attn projections) ----
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # A100+ - ns=3 needs >= 144 KB SMEM
        {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),

    # ---- Large-M (prefill, batched decode): throughput configs ----
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # A100+
        {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),
    triton.Config(  # A100+ - 128x128 doesn't fit L4 at ns=2
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # alt program order for wider-N L2 behaviour
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 4},
        num_warps=8, num_stages=2,
    ),
    triton.Config(  # Hopper/Blackwell - deeper pipeline at 128x128
        {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=3,
    ),

    # ---- XL-M (large prefill on big-SMEM GPUs): H100/Blackwell only ----
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "GROUP_SIZE_M": 8},
        num_warps=8, num_stages=2,
    ),
]


# Autotuner injects BLOCK_M/BLOCK_N/GROUP_SIZE_M + num_warps/num_stages from the
# Config it picks for (M_bucket, N, K). The Config space is a superset of the
# hand-tuned configs, so autotune can at worst tie them. BLOCK_K isn't in the
# space (still passed as constexpr), so all paths bit-match.


def _prune_low_occupancy_configs(configs, nargs, **_kwargs):
    """Drop num_warps<8 candidates at M_bucket >= 2048.

    At large M the high-warp/high-stage configs win, but the gap to a worse pick
    sits in the ~1-2us autotuner noise band, so the coin flip lands wrong about
    half the time (costs ~50us at M=4096). Pruning rather than pinning leaves the
    autotuner free to pick the right large-tile shape per (N, K). Small M_bucket
    legitimately wants fewer warps, so this only triggers for prefill shapes."""
    m_bucket = nargs.get("M_bucket", 0)
    if m_bucket >= 2048:
        filtered = [c for c in configs if c.num_warps >= 8]
        if filtered:
            return filtered
    return configs


_gemm_kernel_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_low_occupancy_configs},
)(_gemm_kernel)


# After the first call for a (M_bucket, N, K, dtype) key, cache the winning
# Config and re-dispatch _gemm_kernel directly. The autotune wrapper's per-call
# dict lookup + config scan + pre_hook is ~30-80us, which dominates at M<=1024 and
# is pure overhead once warmed (the same Config wins every time). Safe because
# Triton's per-signature compiled-kernel cache absorbs the re-dispatch.
_PICKED_CONFIG: dict[tuple[int, int, int, torch.dtype], triton.Config] = {}


def _extract_picked_config(m_bucket: int, N: int, K: int) -> triton.Config | None:
    """Find the Config triton.autotune just picked for (M_bucket, N, K).

    Triton's cache key is (M_bucket, N, K, ...str(dtype)). The dtype suffix length
    varies by version, so match the int prefix and take the first hit - dispatch
    is monomorphic in dtype per key, so the first match is correct. Returns None
    if Triton restructures the cache attr (caller falls back to the autotune
    wrapper)."""
    cache = getattr(_gemm_kernel_autotuned, "cache", None)
    if not cache:
        return None
    for k, cfg in cache.items():
        if (isinstance(k, tuple) and len(k) >= 3
                and k[0] == m_bucket and k[1] == N and k[2] == K):
            return cfg
    return None


def gemm_fxp_run(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    int_bits: int,
    fxp_frac_bits: int,
) -> torch.Tensor:
    # Hot path (once per linear per forward), so skip validation - callers feed
    # known-good tensors and Triton raises on real shape/dtype problems.
    _check_arch(a.dtype)

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c

    scale, inv_scale, qmin, qmax, tl_int_dtype, torch_int_dtype = fxp_constants(
        int_bits, fxp_frac_bits
    )
    io_dtype = _TORCH_TO_TL_FLOAT[a.dtype]
    block_k = _BLOCK_K_BY_DTYPE[a.dtype]
    allow_tf32 = a.dtype == torch.float32

    num_sms = _device_sm_count(a.device)
    target_programs = num_sms * _BLOCKS_PER_SM
    num_k_tiles = triton.cdiv(K, block_k)

    # Tile counts under the split-K kernel's own tiling (_CFG_SPLITK), which is
    # what _resolve_split_k keys and clamps against.
    sk_block_m, sk_block_n, sk_group_size_m, sk_num_warps, sk_num_stages = _CFG_SPLITK
    sk_num_pid_m = triton.cdiv(M, sk_block_m)
    sk_num_tiles_mn = sk_num_pid_m * triton.cdiv(N, sk_block_n)
    scratch_ints = target_programs * _TILE_M_MAX * _TILE_N_MAX
    locks_capacity = _splitk_locks_len(target_programs)
    split_k = _resolve_split_k(
        sk_num_pid_m, num_k_tiles, int_bits, M, N, K, a.dtype, scratch_ints,
        sk_num_tiles_mn, locks_capacity,
    )

    if split_k == 1:
        # Fast path. NUM_SMS is constexpr per GPU so this compiles once. Once the
        # autotuner picks the Config for (M_bucket, N, K, dtype), cache it and
        # re-dispatch _gemm_kernel directly, skipping the wrapper's ~30-80us
        # per-call overhead.
        m_bucket = _bucket_m(M)
        bias_tensor = bias if bias is not None else a
        has_bias = bias is not None

        picked_key = (m_bucket, N, K, a.dtype)
        picked = _PICKED_CONFIG.get(picked_key)

        if picked is None:
            # Cold path: run the autotuner to pick and cache the Config. Use the
            # full target_programs grid since the tile count under its
            # BLOCK_M/BLOCK_N isn't known yet.
            _gemm_kernel_autotuned[(target_programs,)](
                a, b, bias_tensor, c,
                M, N, K, m_bucket,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                SCALE=scale, INV_SCALE=inv_scale,
                QMIN=qmin, QMAX=qmax,
                INT_DTYPE=tl_int_dtype, IO_DTYPE=io_dtype,
                HAS_BIAS=has_bias, ALLOW_TF32=allow_tf32,
                BLOCK_K=block_k, NUM_SMS=target_programs,
            )
            picked = _extract_picked_config(m_bucket, N, K)
            if picked is not None:
                _PICKED_CONFIG[picked_key] = picked
            return c

        # Hot path: raw JIT kernel with the cached Config. Right-size the grid to
        # the tile count (idle programs still cost launch latency). NUM_SMS stays
        # constexpr=target_programs so the stride is fixed and nothing recompiles.
        block_m = picked.kwargs["BLOCK_M"]
        block_n = picked.kwargs["BLOCK_N"]
        num_tiles_mn = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
        grid_size = min(target_programs, num_tiles_mn)
        _gemm_kernel[(grid_size,)](
            a, b, bias_tensor, c,
            M, N, K, m_bucket,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            SCALE=scale, INV_SCALE=inv_scale,
            QMIN=qmin, QMAX=qmax,
            INT_DTYPE=tl_int_dtype, IO_DTYPE=io_dtype,
            HAS_BIAS=has_bias, ALLOW_TF32=allow_tf32,
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE_M=picked.kwargs["GROUP_SIZE_M"],
            NUM_SMS=target_programs,
            num_warps=picked.num_warps,
            num_stages=picked.num_stages,
        )
        return c

    # Persistent scratch (epilogue keeps it clean). The (M, N) view fits since
    # M * N <= num_tiles_mn * 128*128.
    c_int_flat, locks_flat = _get_splitk_scratch(
        a.device, torch_int_dtype, target_programs
    )
    c_int = c_int_flat[: M * N].view(M, N)
    locks = locks_flat[:sk_num_tiles_mn]
    # Right-size the grid to actual tiles (idle programs still cost launch
    # latency). The split-K path is only reached for shapes warmup found it worth,
    # where tiles_mn * split_k stays near num_sms. NUM_SMS stays constexpr=target
    # so the stride is fixed (no recompile on grid changes).
    grid_size = min(target_programs, sk_num_tiles_mn * split_k)
    grid = (grid_size,)
    _gemm_splitk_kernel[grid](
        a,
        b,
        # Triton needs a real tensor; only read when HAS_BIAS.
        bias if bias is not None else a,
        c_int,
        c,
        locks,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_int.stride(0),
        c_int.stride(1),
        c.stride(0),
        c.stride(1),
        num_k_tiles,
        SCALE=scale,
        INV_SCALE=inv_scale,
        QMIN=qmin,
        QMAX=qmax,
        INT_DTYPE=tl_int_dtype,
        IO_DTYPE=io_dtype,
        HAS_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
        BLOCK_M=sk_block_m,
        BLOCK_N=sk_block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=sk_group_size_m,
        SPLIT_K=split_k,
        # NUM_SMS is the persistent loop stride (constexpr); keep it at
        # target_programs even when grid_size shrinks, to avoid per-shape recompile.
        NUM_SMS=target_programs,
        num_warps=sk_num_warps,
        num_stages=sk_num_stages,
    )
    return c


# Warmup-time split-K autotuning. Time each path as a block of back-to-back
# calls, not one synced kernel at a time, because serving runs them back-to-back
# under a CUDA graph where split-K is faster (warm scratch, pipelined launches)
# than single-call timing suggests. Median over a few blocks ignores spikes;
# split must win by a small margin, and since the output is identical either way,
# a tie going to persistent costs nothing.
_AUTOTUNE_WARMUP = 3      # discarded calls before timing
_AUTOTUNE_ITERS = 15      # calls per timed block, so launch cost amortises
_AUTOTUNE_REPS = 3        # blocks per path; compare the medians
_AUTOTUNE_MAX_PID = 16    # probe up to M = 16*BLOCK_M (1024); split rarely helps
                          # past that and the scratch buffer caps it anyway
_AUTOTUNE_MARGIN = 0.02   # split must be at least 2% faster to win


def autotune_split_k(
    weight_native: torch.Tensor, int_bits: int, frac_bits: int
) -> int:
    """Time persistent vs two-way split for this (K, N) weight at each split-K
    row-tile count and cache the faster one in _SPLITK_CHOICE. Call from warmup
    once the persistent configs are compiled. Both paths give the same bits, so
    this only changes speed, never the output. int16 and single-K-tile shapes
    can't split and stay persistent. Returns the number of tile counts routed to
    split-K."""
    global _SPLITK_FORCE
    if int_bits == 16:
        return 0
    K, N = weight_native.shape
    dtype = weight_native.dtype
    device = weight_native.device
    num_k_tiles = triton.cdiv(K, _BLOCK_K_BY_DTYPE[dtype])
    if num_k_tiles < 2:
        return 0
    num_sms = _device_sm_count(device)
    scratch_ints = num_sms * _BLOCKS_PER_SM * _TILE_M_MAX * _TILE_N_MAX
    sk_block_m = _CFG_SPLITK[0]
    paths = (1, _SPLITK_SPLIT)

    routed = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    try:
        for pid_m in range(1, _AUTOTUNE_MAX_PID + 1):
            M = pid_m * sk_block_m
            if M * N > scratch_ints:   # split-2's scratch won't fit beyond here
                break
            a = torch.randn(M, K, device=device, dtype=dtype)
            # time each path as a block of back-to-back calls (like serving),
            # take the median over a few blocks
            t = {}
            for force in paths:
                _SPLITK_FORCE = force
                for _ in range(_AUTOTUNE_WARMUP):   # settle and JIT, discarded
                    gemm_fxp_run(a, weight_native, None, int_bits, frac_bits)
                torch.cuda.synchronize(device)
                per_call = []
                for _ in range(_AUTOTUNE_REPS):
                    start.record()
                    for _ in range(_AUTOTUNE_ITERS):
                        gemm_fxp_run(a, weight_native, None, int_bits, frac_bits)
                    end.record()
                    end.synchronize()
                    per_call.append(start.elapsed_time(end) / _AUTOTUNE_ITERS)
                t[force] = statistics.median(per_call)
            use_split = t[_SPLITK_SPLIT] < t[1] * (1.0 - _AUTOTUNE_MARGIN)
            _SPLITK_CHOICE[(pid_m, N, K, dtype, int_bits)] = (
                _SPLITK_SPLIT if use_split else 1
            )
            routed += int(use_split)
    finally:
        _SPLITK_FORCE = None
    return routed
