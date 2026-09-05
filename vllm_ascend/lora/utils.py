import importlib

import torch
import vllm

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)

# Full-rank A after TP all-gather (fully-sharded LoRA). Eager set_lora only.
# Used only by add_lora_linear (non-mcp layers). Do not divert _mcp_apply:
# official shrink → all_gather → expand is what this stack's ACL graphs keep.
# e389704 patched _mcp_apply to return fused y; chat identity became the base model.
FULL_RANK_LORA_A_ATTR = "_full_rank_lora_a"
_LINEAR_LORA_HOOKS_INSTALLED = False
_SKIP_NAME_PARTS = ("MoE", "Embedding", "Logits", "Vocab")


def refresh_all_lora_classes():
    ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
    install_linear_lora_hooks()


def gather_sharded_lora_a(layer) -> None:
    """All-gather LoRA A on the rank dim so fused kernels see the full rank."""
    if torch.compiler.is_compiling():
        return
    stacked = getattr(layer, "lora_a_stacked", None)
    bstack = getattr(layer, "lora_b_stacked", None)
    if not isinstance(stacked, (tuple, list)) or not isinstance(bstack, (tuple, list)):
        return
    if not stacked or not bstack or not torch.is_tensor(stacked[0]):
        return
    r_a = stacked[0].size(-2)
    r_b = bstack[0].size(-1)
    if r_a >= r_b:
        setattr(layer, FULL_RANK_LORA_A_ATTR, tuple(stacked))
        return
    try:
        from vllm.distributed import tensor_model_parallel_all_gather
    except ImportError:
        return
    if tensor_model_parallel_all_gather is None:
        return
    full = []
    for src in stacked:
        t = src.transpose(-1, -2).contiguous()
        g = tensor_model_parallel_all_gather(t, dim=-1)
        full.append(g.transpose(-1, -2).contiguous())
    setattr(layer, FULL_RANK_LORA_A_ATTR, tuple(full))


def _wrap_after_gather(orig):
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        gather_sharded_lora_a(self)
        return out

    return wrapped


def _iter_linear_lora_classes():
    seen: set[type] = set()
    classes = list(getattr(vllm.lora.utils, "_all_lora_classes", ()))
    extra_targets = (
        ("vllm.lora.layers.base_linear", "BaseLinearLayerWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "ColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedQKVParallelLinearWithLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "ColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "QKVParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.column_parallel_linear", "MergedQKVParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers.row_parallel_linear", "RowParallelLinearWithLoRA"),
        ("vllm.lora.layers.row_parallel_linear", "RowParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "BaseLinearLayerWithLoRA"),
        ("vllm.lora.layers", "MergedColumnParallelLinearWithLoRA"),
        ("vllm.lora.layers", "MergedQKVParallelLinearWithLoRA"),
        ("vllm.lora.layers", "ColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "MergedColumnParallelLinearWithShardedLoRA"),
        ("vllm.lora.layers", "MergedQKVParallelLinearWithShardedLoRA"),
    )
    for mod_name, attr in extra_targets:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, attr, None)
        if isinstance(cls, type):
            classes.append(cls)
    for cls in classes:
        if not isinstance(cls, type) or cls in seen:
            continue
        name = getattr(cls, "__name__", "")
        if any(part in name for part in _SKIP_NAME_PARTS):
            continue
        seen.add(cls)
        yield cls


def install_linear_lora_hooks():
    """Gather sharded A at set_lora. Do not patch apply/_mcp_apply/_apply_lora_to_output."""
    global _LINEAR_LORA_HOOKS_INSTALLED
    for cls in _iter_linear_lora_classes():
        if getattr(cls, "_fused_lora_hooked", False):
            continue
        for name in ("create_lora_weights", "set_lora", "reset_lora", "set_mapping"):
            orig = getattr(cls, name, None)
            if orig is None or not callable(orig):
                continue
            setattr(cls, name, _wrap_after_gather(orig))
        cls._fused_lora_hooked = True  # type: ignore[attr-defined]
    _LINEAR_LORA_HOOKS_INSTALLED = True
