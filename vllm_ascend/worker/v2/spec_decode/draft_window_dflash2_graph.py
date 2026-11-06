# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""DFlash2's fixed selector inputs and native-resident Graph source contract.

Only Host metadata is inspected here. Dispatch, request reordering and metadata
preparation remain outside capture. The inherited DFlash forward captures the
convolution, candidate TopK, selector, NPU walk and probability-cache writes;
this module neither clones those algorithms nor creates another Graph owner.
"""

from __future__ import annotations

from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("DFlash2 Graph: " + message)


def validate_dflash2_graph_source(owner: Any, group: Any = None) -> None:
    """Qualify actual TP2/K7 Graph inputs after loading, before capture.

    Window ON calls this from its read-source admission; OFF calls it from
    manager construction. Fixed constructors own selector/conv probability
    storage; set_attn has already qualified the actual native consumers.
    """
    vllm_config = owner.vllm_config
    config = owner.speculative_config
    parallel = vllm_config.parallel_config
    draft_parallel = config.draft_parallel_config
    topology = tuple(
        getattr(parallel, name, None)
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    )
    _require(
        topology == (2, 1, 1, 1, 1) and getattr(draft_parallel, "tensor_parallel_size", None) == 2,
        "first Graph qualification requires actual Target/Draft TP2, DP1/PP1/PCP1/DCP1",
    )
    _require(
        config.num_speculative_tokens == 7
        and not config.uses_dynamic_speculative_decoding()
        and not config.enable_adaptive_verification,
        "first Graph qualification requires static nonadaptive K7/Q8",
    )
    additional = vllm_config.additional_config or {}
    _require(
        additional.get("enable_dsa_cp", False) is False
        and additional.get("enable_flashcomm1", False) is False
        and getattr(parallel, "use_sequence_parallel_moe", None) is False
        and getattr(draft_parallel, "use_sequence_parallel_moe", None) is False,
        "first Graph qualification excludes FlashComm, SP and legacy DSA-CP",
    )
    _require(
        tuple(owner.draft_model_config.architectures) == ("DFlash2DraftModel",), "actual DFlash2 architecture required"
    )
    _require(owner.use_fp64_gumbel is False, "the NPU selector does not implement fp64 Gumbel")
    # The fixed DFlash2 constructors use this same config for Q=K+1,
    # selector/sampler top-k, per-layer convolution blocks and tensor storage.
    _require(0 < owner.top_k <= owner.vocab_size, "selector top-k exceeds the actual vocabulary")
    _require(
        getattr(owner, "_dflash2_full_kv_reference", False) is True,
        "the actual DFlash2 full-retention loader contract is missing",
    )
    # Real set_attn qualified loaded implementations and native mask bindings
    # in its native-attention pass before this manager/source is constructed.

    if group is not None:
        causal = owner._group_causal
        _require(
            (causal if type(causal) is bool else causal.get(group.gid)) is False,
            "DFlash2 Graph source requires actual noncausal attention",
        )
        # Inventory compares the native band against the loaded consumer;
        # Full KV retention itself establishes no attention range.
        if group.native_window is not None:
            window = owner._draft_window_policy
            _require(
                window + group.table_column_span - 1 + owner.num_query_per_req <= group.native_window,
                "Window + block - 1 + query width exceeds the original noncausal native band",
            )
