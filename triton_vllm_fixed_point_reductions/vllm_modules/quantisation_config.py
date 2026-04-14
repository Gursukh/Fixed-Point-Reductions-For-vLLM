from typing import Any, Dict, List, Optional, Sequence
import os
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.quantization import register_quantization_config

from ..fixed_point_kernels.fixed_point import flp_2_fxp, fxp_to_flp


DEBUG_TIMING = os.getenv("VLLM_FXP_DEBUG_TIMING", "0") == "1"

DEFAULT_FRAC_BITS = 16


@triton.jit
def _gemm_fxp_launcher(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    D_CHUNK: tl.constexpr,
    FRAC_BITS: tl.constexpr,
):
    """
    Thin wrapper around ``gemm_fxp_kernel`` that decouples the inner
    D_CHUNK from BLOCK_SIZE_K so the intermediate ``[ROWS, D_CHUNK, COLS]``
    product tile stays under Triton's 1M-element limit even when the
    reduction dim is large (Qwen3-0.6B has K=1024).
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_m < M
    col_mask = offs_n < N

    a_row_ptrs = a_ptr + offs_m * stride_am
    b_col_ptrs = b_ptr + offs_n * stride_bn

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.int32)

    for k_start in tl.range(0, BLOCK_SIZE_K, D_CHUNK):
        k_offs = k_start + tl.arange(0, D_CHUNK)
        k_valid = k_offs < K

        a = tl.load(
            a_row_ptrs[:, None] + k_offs[None, :] * stride_ak,
            mask=row_mask[:, None] & k_valid[None, :],
            other=0.0,
        ).to(tl.float32)

        b = tl.load(
            b_col_ptrs[None, :] + k_offs[:, None] * stride_bk,
            mask=k_valid[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Inner partial dot-product in fp32 uses tensor cores (fast compile,
        # fast runtime). The cross-chunk reduction is then done in Q-format
        # fixed point, which is what actually buys order-invariance across
        # different launch shapes.
        partial = tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        acc += flp_2_fxp(partial, FRAC_BITS, tl.int32)

    c = fxp_to_flp(acc, FRAC_BITS, tl.float32)

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def _launch_gemm_fxp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run c = a @ b via the fixed-point Triton GEMM kernel (fp32 in/out)."""
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]

    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Fixed M/N tiles → per-element result is batch-shape invariant.
    # BLOCK_K covers the full reduction (kernel uses Lk as mask only).
    # D_CHUNK is the inner loop step; the intermediate product tile is
    # [BLOCK_M, D_CHUNK, BLOCK_N] and must fit Triton's 1M-element limit.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(max(K, 1))
    D_CHUNK = 32  # 64*32*64 = 131072 ≤ 1M

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _gemm_fxp_launcher[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        D_CHUNK=D_CHUNK,
        FRAC_BITS=DEFAULT_FRAC_BITS,
    )
    return c

@register_quantization_config("fixed_point_det")
class FixedPointConfig(QuantizationConfig):
    """
    Configuration for bitwise-deterministic fixed-point linear layers.
    """

    def __init__(self, frac_bits: int = DEFAULT_FRAC_BITS) -> None:
        self.frac_bits = frac_bits

    def __repr__(self) -> str:
        return f"FixedPointConfig(frac_bits={self.frac_bits})"

    # ----- registry integration -----

    @classmethod
    def get_name(cls) -> str:
        return "fixed_point_det"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM70+ (Volta) — int32 tensor core ops available
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        # No on-disk config file needed — this isn't a checkpoint format.
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FixedPointConfig":
        frac_bits = config.get("frac_bits", DEFAULT_FRAC_BITS)
        return cls(frac_bits=frac_bits)

    # ----- method dispatch -----

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return FixedPointLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FixedPointLinearMethod(QuantizeMethodBase):
    """
    Linear method that stores weights in the original dtype but executes
    matmuls via a deterministic fixed-point Triton GEMM.
    """

    def __init__(self, config: FixedPointConfig) -> None:
        self.config = config
        self._debug_calls = 0

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: Sequence[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """
        Create the weight parameter in the original dtype so vLLM's
        checkpoint loader can fill it normally.
        """
        total_output_size = sum(output_partition_sizes)

        weight = ModelWeightParameter(
            data=torch.empty(
                total_output_size,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

        # Store partition info for per-partition scale computation
        layer.output_partition_sizes = list(output_partition_sizes)
        layer.input_size_per_partition = input_size_per_partition

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """No-op: the Triton GEMM quantises fp32 weights on the fly."""
        return

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic forward pass via the fixed-point Triton GEMM."""
        t0 = time.perf_counter() if DEBUG_TIMING else 0.0
        orig_dtype = x.dtype

        x2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
        w = layer.weight.data.to(torch.float32)  # (out, in)

        out = _launch_gemm_fxp(x2d, w.t())  # (M, out)
        out = out.view(*x.shape[:-1], out.shape[-1])

        if bias is not None:
            out = out + bias.to(torch.float32)

        if DEBUG_TIMING:
            self._debug_calls += 1
            # Print first few calls and then periodically to avoid flooding.
            if self._debug_calls <= 10 or self._debug_calls % 100 == 0:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[fxp-linear] call={self._debug_calls} "
                    f"in={tuple(x.shape)} w={tuple(layer.weight.shape)} "
                    f"out={tuple(out.shape)} ms={dt_ms:.3f}"
                )

        return out.to(orig_dtype)