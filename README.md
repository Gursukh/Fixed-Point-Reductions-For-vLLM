# Fixed-Point Reductions (FxPR) for vLLM

FxPR for vLLM allows for deterministic inference by removing the non-determinism introduced by floating-point non-associativity. We provide a patch for vLLM that implement the same mathematical operations as the original vLLM kernels, but execute any associativity-dependent reduction in fixed-point arithmetic. Integer addition is associative, so the result is bitwise-identical regardless of how the work is split across SMs, warps, or KV-cache splits.

## How it works

Every reduction (sum, dot-product, softmax normaliser, RMS-norm squared-sum) follows the same pattern:

1. Cast float operands to a signed fixed-point integer via `float_to_fixed(x, frac_bits, int_dtype)` — round-to-nearest-even with saturation.
2. Do the reduction as an integer sum.
3. Cast the integer accumulator back to float via `fixed_to_float`.

Because the intermediate accumulator is an integer with a fixed scale, reordering the additions cannot change the result. 


## Installation

```bash
pip install git+https://github.com/Gursukh/vLLM-Fixed-Point-Reductions.git
```

## Usage


```python
from vllm import LLM
from vllm_fixed_point_reductions.register import register
register()

llm = LLM(
    ...
    quantization="fixed_point_det"
    attention_backend="CUSTOM",
)
```

### Runtime configuration

See [config.py](fxpr_vllm/vllm_modules/config.py)

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_FXP_FRAC_BITS` | `16` | Number of fractional bits in the Q-format (higher = finer resolution, smaller dynamic range). |
| `VLLM_FXP_INT_BITS` | `32` | Accumulator width. One of `16`, `32`, `64`. |
| `VLLM_FXP_NUM_KV_SPLITS` | `8` | Number of KV splits for the decode attention kernel. |