# Fixed-Point Reductions for vLLM

A drop-in plugin that makes vLLM's floating-point reductions bitwise
reproducible. The idea is deliberately unglamorous: cast to a signed
fixed-point integer before reducing, accumulate in integer space, then cast
back. Integer addition is associative and commutative, so the result no longer
depends on how the reduction is partitioned across SMs, warps, or KV splits.
The same logits come out regardless of batch composition, tensor-parallel
degree, or kernel tiling — the usual sources of run-to-run drift in LLM
inference.

The plugin replaces the reductions that matter for determinism — the GEMMs
(including the LM head), RMSNorm, and attention — with fixed-point variants,
and registers them with vLLM through its standard quantization and attention
backend hooks.

## Installation

```bash
pip install git+https://github.com/Gursukh/Fixed-Point-Reductions-For-vLLM.git
```

Requirements: Python ≥ 3.12, `torch>=2.6`, `triton>=3.0`, and `vllm==0.20.2`.
The fixed-point GEMM uses tensor cores and requires compute capability sm_75+
(Turing) for fp16; bf16 and fp32 paths additionally require sm_80+ (Ampere).

## Usage

```python
from vllm import LLM

llm = LLM(
    model=...,
    quantization="fixedpoint",
    attention_backend="CUSTOM",
)
```

Registration happens automatically through the `vllm.general_plugins` entry
point — there is no explicit initialization call. Setting `quantization`
selects the fixed-point quant config (which routes the GEMMs and LM head), and
`attention_backend="CUSTOM"` selects the deterministic attention backend. The
RMSNorm patch is applied at registration time and is on by default.

### Configuration

All behavior is configured through environment variables read once at process
start; see [`config.py`](fxpr_vllm/config.py). Defaults are chosen so that the
plugin is fully deterministic out of the box.

| Variable                         | Default | Allowed       | Effect |
| -------------------------------- | ------- | ------------- | ------ |
| `FXPR_INT_BITS`                  | `32`    | `16`/`32`/`64`| Width of the integer accumulator. |
| `FXPR_FRAC_BITS`                 | `16`    | `8`/`16`/`32` | Fractional bits in the Q-format. More bits give finer resolution at the cost of dynamic range. |
| `FXPR_DISABLE_RMS_NORM`          | unset   | `1`           | Skip the `DeterministicRMSNorm` patch and leave vLLM's stock RMSNorm in place. |
| `FXPR_DISABLE_LM_HEAD`           | unset   | `1`           | Skip the fixed-point LM-head matmul. |
| `FXPR_DISABLE_ATTENTION_WARMUP`  | unset   | `1`           | Skip autotuning warmup for the attention kernels. |
| `FXPR_DISABLE_GEMM_WARMUP`       | unset   | `1`           | Skip autotuning warmup for the GEMM kernels. |
| `FXPR_DISABLE_RMS_NORM_WARMUP`   | unset   | `1`           | Skip autotuning warmup for the RMSNorm kernels. |

The `int_bits` / `frac_bits` pair sets the representable range. Q16.16 — i.e.
`FXPR_FRAC_BITS=16` in a 32-bit accumulator — covers roughly ±32K in the
original units before saturation. Raise `FXPR_INT_BITS` to 64 if your
activations have a wider distribution.

## Module layout

```
fxpr_vllm/
  _lib.py                torch.library schemas + CUDA impls (dispatch to Triton)
  library_ops.py         register_fake meta impls for Dynamo, plus op aliases
  register.py            vLLM plugin entry point
  monkey_patches.py      patches for the parts vLLM doesn't expose cleanly
  config.py              environment-variable runtime configuration
  warmup.py              autotuning warmup for the Triton kernels
  quantisation_config.py fixedpoint quant config (GEMM + LM head)
  attention_backend.py   CUSTOM attention backend
  rms_norm.py            DeterministicRMSNorm module
  _triton/
    fxp.py               float <-> fixed device helpers (rint, clamp, cast)
    casts.py             float_to_fixed / fixed_to_float
    rms_norm.py          RMSNorm + fused-residual variant
    gemm.py              tensor-core GEMM with fixed-point inter-tile accumulation
    attention.py         unified prefill + decode, paged KV, split-K
```

## Tests

```bash
pytest
```

The suite lives in `fxpr_vllm/_tests/` (configured as the pytest `testpaths`)
and requires CUDA. The bf16/fp32 GEMM tests require sm_80+ (Ampere or newer)
and skip on older hardware.
