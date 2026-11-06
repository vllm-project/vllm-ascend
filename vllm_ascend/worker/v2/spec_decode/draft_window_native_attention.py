# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Host-only startup inventory of the Draft model's native attention.

KV retention is not an attention contract. In particular, DFlash2 keeps Full
KV backing while its actual consumers retain their native sliding windows.
This inventory neither creates a Window owner nor enables a read provider.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from vllm.logger import logger

_MISSING = object()
_DRAFT_ARCHITECTURES = frozenset(
    {
        "DFlashDraftModel",
        "DFlash2DraftModel",
        "Qwen3DSparkModel",
        "Qwen3OmniDSparkModel",
        "LlamaForCausalLMEagle3",
        "Glm5DSparkForCausalLM",
    }
)


def inspect_draft_native_attention(
    speculator: Any, *, dflash2_graph: bool = False
) -> tuple[str, tuple[int, ...], tuple[str, ...]]:
    """Inspect loaded Draft consumers, after the real loader and ``set_attn``.

    Normalized architecture/layer count bounds the inventory. Native windows
    come from the executing implementations, never a KV storage spec or a
    second interpretation of the original model config/defaults.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

    normalized = speculator.draft_model_config.hf_config
    architectures: Sequence[str] = getattr(normalized, "architectures", ()) or ()
    count = getattr(normalized, "num_hidden_layers", None)
    issues = []
    known = len(architectures) == 1 and architectures[0] in _DRAFT_ARCHITECTURES
    if not known or type(count) is not int or count <= 0:
        issues.append("normalized Draft architecture or layer count has no supported implementation contract")
    context = speculator.vllm_config.compilation_config.static_forward_context
    declared_causal = getattr(speculator, "_group_causal", _MISSING)
    layers: dict[str, int | None] = {}
    for gid, group_set in enumerate(speculator.attn_groups):
        causal = declared_causal.get(gid, _MISSING) if isinstance(declared_causal, Mapping) else declared_causal
        for group in group_set:
            for name in group.layer_names:
                if name in layers:
                    issues.append(f"Draft consumer {name} is registered more than once")
                    continue
                impl = getattr(context.get(name), "impl", None)
                if type(impl) not in (AscendAttentionBackendImpl, AscendMLAImpl):
                    identity = f"{type(impl).__module__}.{type(impl).__qualname__}"
                    issues.append(f"Draft consumer {name} has unknown native attention implementation {identity}")
                    continue
                if dflash2_graph and (
                    type(impl) is not AscendAttentionBackendImpl or impl.sinks is not None or impl.use_bnsd_kv_cache
                ):
                    issues.append(f"DFlash2 Graph consumer {name} requires the actual unpacked dense FIA without sinks")
                    continue
                value = getattr(impl, "sliding_window", _MISSING)
                if type(impl) is AscendMLAImpl:
                    # This dense implementation has no native window operand.
                    value = None
                if value is _MISSING:
                    issues.append(f"Draft consumer {name} has no native window descriptor")
                    continue
                if value is not None and (type(value) is not int or value <= 0):
                    issues.append(f"Draft consumer {name}: native sliding_window must be a positive integer or None")
                    continue
                layers[name] = value
                if value is not None and type(causal) is not bool:
                    issues.append(f"Draft consumer {name} has no actual native causality descriptor")
    # The loader already identifies the Draft layer set, including any Target
    # layer offset. Use that registration instead of parsing layer-name strings.
    declared = speculator.draft_attn_layer_names
    complete = bool(layers) and known and type(count) is int and len(layers) == count
    if not complete or set(layers) != set(declared):
        issues.append("complete supported Draft architecture and actual consumer inventory are required")
    windows = tuple(sorted({value for value in layers.values() if value is not None}))
    kind = "UNKNOWN"
    if not issues:
        kind = "FULL" if not windows else "MIXED" if None in layers.values() else "SWA"
    return kind, windows, tuple(dict.fromkeys(issues))


def report_owner_draft_native_attention(owner: Any) -> None:
    """Startup-only summary of the actual speculator's loaded attention.

    The fixed loader owns original/normalized configuration. No config file,
    runtime, metadata tensor or completion object is read or allocated here.
    """
    window = getattr(owner, "_draft_window_policy", None) or getattr(owner, "draft_window_policy", None)
    speculative = owner.speculative_config
    eager = speculative.enforce_eager is True or getattr(owner.draft_model_config, "enforce_eager", False) is True
    # All real callers run here from set_attn, before the fixed runner calls
    # init_cudagraph_manager. Report the Draft request, not Target Graph mode
    # or a hypothetical already-created manager.
    graph_requested = not eager
    architectures = getattr(owner.draft_model_config.hf_config, "architectures", ()) or ()
    new_dflash2_graph = graph_requested and tuple(architectures) == ("DFlash2DraftModel",)
    kind, windows, issues = inspect_draft_native_attention(owner, dflash2_graph=new_dflash2_graph)
    logger.info_once(
        "Draft native attention=%s; native windows=%s; extra Draft window=%s; "
        "Draft Graph requested=%s; complete actual inventory=%s; issues=%s. "
        "Full KV retention does not imply Full attention.",
        kind,
        windows,
        window,
        graph_requested,
        kind != "UNKNOWN",
        issues,
        scope="global",
    )
    # Only observed/certified native consumers establish this warning.
    narrowed = tuple(value for value in windows if window < value) if window is not None and kind != "UNKNOWN" else ()
    if narrowed:
        logger.warning_once(
            "The Draft model contains native SWA layers with windows %s. "
            "The additional draft_window_size=%s narrows those layers and may reduce "
            "acceptance or throughput. Prefer the native window by default and "
            "enable an additional window only after measuring your workload.",
            narrowed,
            window,
            scope="global",
        )
    # Detection must not turn an original Window-OFF Graph route (for example
    # built-in MTP) into a newly rejected configuration. Additional Window
    # owners always need the contract; DFlash2 Graph is itself new admission
    # in this change and needs it even when the additional Window is OFF.
    if (window is not None or new_dflash2_graph) and kind == "UNKNOWN":
        raise NotImplementedError("Draft Window/Graph requires a known native attention contract: " + "; ".join(issues))
