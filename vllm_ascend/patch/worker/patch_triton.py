# Lazy import to avoid circular dependency issues during conftest loading
# The patches will be applied when vllm actually tries to use these modules
import sys
from importlib.abc import MetaPathFinder


_patches_applied = False


def _apply_triton_patches():
    """Apply triton-related patches lazily to avoid import-time circular dependencies."""
    global _patches_applied
    if _patches_applied:
        return

    try:
        from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
        from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
        from vllm_ascend.ops.triton.fla.sigmoid_gating import \
            fused_recurrent_gated_delta_rule_fwd_kernel
        from vllm_ascend.ops.triton.mamba.causal_conv1d import (
            causal_conv1d_fn, causal_conv1d_update_npu)

        import vllm.model_executor.layers.mamba.ops.causal_conv1d
        vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
        vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
        vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = fused_recurrent_gated_delta_rule_fwd_kernel
        vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
        vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
        _patches_applied = True
    except (ImportError, AttributeError):
        pass


class _TritonPatchLoader(MetaPathFinder):
    """Meta path finder that applies triton patches when vllm.model_executor is imported."""

    _trigger_modules = {
        'vllm.model_executor.layers.mamba',
        'vllm.model_executor.layers.fla',
    }

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        # Check if this is one of the modules we want to patch
        for trigger in self._trigger_modules:
            if fullname.startswith(trigger):
                _apply_triton_patches()
                # Remove ourselves from meta_path after first trigger
                if self in sys.meta_path:
                    sys.meta_path.remove(self)
                break
        return None  # Let default import machinery handle the actual import


# Install the import hook
sys.meta_path.insert(0, _TritonPatchLoader())
