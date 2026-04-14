import logging
import os
import sys
import time


logger = logging.getLogger("vllm_deterministic")

_registered = False


def _debug_enabled() -> bool:
    return (
        os.getenv("FXP_DEBUG_TIMING", "0") == "1"
        or os.getenv("VLLM_FXP_DEBUG_TIMING", "0") == "1"
        or os.getenv("FXP_DEBUG_POST", "0") == "1"
        or os.getenv("VLLM_FXP_DEBUG_POST", "0") == "1"
    )


def _cuda_sync_if_available() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _patch_gpu_post_timing() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_fxp_post_timing_patched", False):
        return

    orig_execute_model = GPUModelRunner.execute_model
    orig_sample_tokens = GPUModelRunner.sample_tokens
    orig_bookkeeping_sync = GPUModelRunner._bookkeeping_sync

    def execute_model_with_timing(self, *args, **kwargs):
        orig_compute_logits = getattr(self.model, "compute_logits", None)

        if callable(orig_compute_logits):
            def timed_compute_logits(hidden_states, *c_args, **c_kwargs):
                hs_shape = tuple(hidden_states.shape) if hasattr(hidden_states, "shape") else "unknown"
                _cuda_sync_if_available()
                t_logits = time.perf_counter()
                out = orig_compute_logits(hidden_states, *c_args, **c_kwargs)
                _cuda_sync_if_available()
                dt_ms = (time.perf_counter() - t_logits) * 1000.0
                print(
                    f"[fxp-post] compute_logits hidden={hs_shape} ms={dt_ms:.3f}"
                )
                return out

            self.model.compute_logits = timed_compute_logits

        _cuda_sync_if_available()
        t0 = time.perf_counter()
        try:
            return orig_execute_model(self, *args, **kwargs)
        finally:
            _cuda_sync_if_available()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[fxp-post] execute_model total_ms={dt_ms:.3f}")
            if callable(orig_compute_logits):
                self.model.compute_logits = orig_compute_logits

    def sample_tokens_with_timing(self, *args, **kwargs):
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        out = orig_sample_tokens(self, *args, **kwargs)
        _cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        sampled = "none"
        if hasattr(out, "sampled_token_ids") and out.sampled_token_ids is not None:
            sampled = str(tuple(out.sampled_token_ids.shape))

        print(f"[fxp-post] sample_tokens sampled={sampled} ms={dt_ms:.3f}")
        return out

    def bookkeeping_sync_with_timing(self, *args, **kwargs):
        logits = args[2] if len(args) > 2 else kwargs.get("logits")
        logits_shape = tuple(logits.shape) if hasattr(logits, "shape") else "unknown"

        _cuda_sync_if_available()
        t0 = time.perf_counter()
        out = orig_bookkeeping_sync(self, *args, **kwargs)
        _cuda_sync_if_available()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[fxp-post] bookkeeping logits={logits_shape} ms={dt_ms:.3f}"
        )
        return out

    GPUModelRunner.execute_model = execute_model_with_timing
    GPUModelRunner.sample_tokens = sample_tokens_with_timing
    GPUModelRunner._bookkeeping_sync = bookkeeping_sync_with_timing
    GPUModelRunner._fxp_post_timing_patched = True


def register() -> None:
    """
    Called once by vLLM's plugin loader at startup.
    Idempotent — safe to call multiple times.
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("vllm-deterministic: registering components")

    # ------------------------------------------------------------------
    # 1. Deterministic RMSNorm (CustomOp)
    # ------------------------------------------------------------------
    # Importing the module registers the custom op name. We also monkey patch
    # the class binding so direct RMSNorm(...) constructions use our class.
    from .rms_norm_fxp import DeterministicRMSNorm
    import vllm.model_executor.layers.layernorm as layernorm_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


    layernorm_mod.RMSNorm = DeterministicRMSNorm

    # If model modules were already imported before plugin registration,
    # patch their local RMSNorm bindings too.
    patched_modules = 0
    for mod_name in (
        "vllm.model_executor.models.qwen2",
        "vllm.model_executor.models.qwen3",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "RMSNorm"):
            setattr(mod, "RMSNorm", DeterministicRMSNorm)
            patched_modules += 1

    logger.info(
        "  ✓ RMSNorm patched (layernorm.RMSNorm=%s, preloaded_model_modules=%d)",
        layernorm_mod.RMSNorm.__name__,
        patched_modules,
    )

    # ------------------------------------------------------------------
    # 2. Fixed-point quantisation config
    # ------------------------------------------------------------------
    from .quantisation_config import FixedPointConfig  # noqa: F401

    
    # ------------------------------------------------------------------
    # 3. Deterministic attention backend
    # ------------------------------------------------------------------
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    from .attention_backend import DeterministicAttentionBackend

    backend_path = (
        f"{DeterministicAttentionBackend.__module__}."
        f"{DeterministicAttentionBackend.__qualname__}"
    )

    register_backend(
        AttentionBackendEnum.TRITON_ATTN,
        class_path=backend_path,
    )
    logger.info(
        "  ✓ Attention backend registered "
        "(overrides TRITON_ATTN → %s)",
        backend_path,
    )

    if _debug_enabled():
        _patch_gpu_post_timing()
        logger.info("  ✓ GPU post-layer timing hooks enabled")

    # # ------------------------------------------------------------------
    # # 4. Deterministic TP all-reduce (only if TP > 1)
    # # ------------------------------------------------------------------
    # tp_size = int(os.environ.get("VLLM_TP_SIZE", "1"))

    # # Also check the command-line style env var
    # if tp_size <= 1:
    #     tp_size = int(
    #         os.environ.get("TENSOR_PARALLEL_SIZE", "1")
    #     )

    # if tp_size > 1:
    #     from .a import patch_tp_allreduce

    #     patch_tp_allreduce()
    #     logger.info("  ✓ TP all-reduce patched (TP=%d)", tp_size)
    # else:
    #     logger.info("  - TP all-reduce skipped (TP=1)")

    # ------------------------------------------------------------------
    # 5. Deterministic sampling log-softmax
    # ------------------------------------------------------------------
    from .sampling import patch_sampler_log_softmax

    patch_sampler_log_softmax()
    logger.info("  ✓ Sampler log-softmax patched")
