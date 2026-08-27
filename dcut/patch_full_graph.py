# SPDX-License-Identifier: Apache-2.0
"""FULL_DECODE_ONLY multi-shape support for D-Cut.

D-Cut turns the verifier batch from a uniform ``num_spec + 1`` rectangle into
a ragged decode batch.  Stock vLLM only registers uniform FULL keys for
``FULL_DECODE_ONLY``, so every useful trimmed step otherwise falls back to
eager execution.

This module keeps the stock uniform keys (used by untrimmed verification and
the draft model), adds non-uniform FULL keys for the same token buckets, and
isolates Ascend full-attention graph parameters by ``BatchDescriptor``.  The
latter is required because uniform and ragged graphs can have the same padded
token count but capture different task-group handles and tensor addresses.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from dataclasses import replace
from typing import Any

import torch

from .globals import ENV_CONFIG, ENV_FULL_DECODE_ONLY, logger
from .truncate import _dcut_has_prefill
from .utils import _supports_adaptive_verify


def _dcut_full_decode_multishape_enabled(vllm_config=None) -> bool:
    """Return whether D-Cut should extend FULL_DECODE_ONLY graph dispatch."""
    if not os.environ.get(ENV_CONFIG) or os.environ.get(ENV_FULL_DECODE_ONLY):
        return False
    if vllm_config is None:
        return True
    return _supports_adaptive_verify(
        getattr(vllm_config, "speculative_config", None)
    )


def _dcut_ragged_full_request_capacity(
    num_tokens: int,
    max_num_seqs: int,
) -> int:
    """Return the fixed request capacity for one ragged token bucket.

    A verifier request always contributes at least its anchor token, so a
    ``num_tokens`` graph can never contain more than ``num_tokens`` requests.
    Keeping this capacity a pure function of the token bucket lets all live
    batch sizes reuse one FULL graph without introducing a second graph axis.
    """
    num_tokens = int(num_tokens)
    max_num_seqs = int(max_num_seqs)
    if num_tokens <= 0 or max_num_seqs <= 0:
        raise ValueError(
            "D-Cut ragged FULL capacity requires positive token and request "
            f"limits, got num_tokens={num_tokens}, max_num_seqs={max_num_seqs}"
        )
    return min(num_tokens, max_num_seqs)


class _DescriptorGraphParamMap(MutableMapping[int, Any]):
    """Map graph parameters by ``(num_tokens, BatchDescriptor)``.

    Ascend's graph-parameter APIs index these mappings with an integer token
    count.  Resolve the active descriptor from ForwardContext so existing
    attention implementations need no changes.  Iteration intentionally still
    exposes integer capture sizes for ``weak_ref_workspaces`` compatibility.
    """

    def __init__(self, initial: dict[int, Any], *, list_values: bool) -> None:
        self._sizes = tuple(initial)
        self._list_values = list_values
        self._values: dict[tuple[int, Any], Any] = {}

    @staticmethod
    def _descriptor():
        try:
            from vllm.forward_context import get_forward_context

            context = get_forward_context()
            return getattr(context, "batch_descriptor", None)
        except Exception:
            return None

    def _key(self, num_tokens: int) -> tuple[int, Any]:
        return int(num_tokens), self._descriptor()

    def __getitem__(self, num_tokens: int) -> Any:
        key = self._key(num_tokens)
        if key not in self._values:
            self._values[key] = [] if self._list_values else None
        return self._values[key]

    def __setitem__(self, num_tokens: int, value: Any) -> None:
        self._values[self._key(num_tokens)] = value

    def __delitem__(self, num_tokens: int) -> None:
        del self._values[self._key(num_tokens)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._sizes)

    def __len__(self) -> int:
        return len(self._sizes)


def _dcut_rebind_live_fia_block_tables(
    captured_params,
    attn_metadata,
) -> None:
    """Rebind captured FIA task params to the runtime live-row block table."""
    if not isinstance(attn_metadata, dict) or not attn_metadata:
        return
    metadata_keys = tuple(attn_metadata)
    for index, param in enumerate(tuple(captured_params)):
        if not isinstance(param, tuple) or len(param) < 4:
            continue
        layer_name = param[-1] if isinstance(param[-1], str) else None
        metadata_key = (
            layer_name
            if layer_name is not None and layer_name in attn_metadata
            else metadata_keys[index % len(metadata_keys)]
        )
        live_block_tables = getattr(
            attn_metadata[metadata_key], "block_tables", None
        )
        if live_block_tables is None:
            continue
        captured_params[index] = (
            *param[:3],
            live_block_tables,
            *param[4:],
        )


def _patch_live_fia_graph_params() -> None:
    """Refresh sink-FIA block tables when ragged FULL request rows change."""
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.attention.utils import using_paged_attention
    from vllm_ascend.compilation.acl_graph import get_graph_params

    impl_cls = AscendAttentionBackend.get_impl_cls()
    if getattr(impl_cls, "_dcut_live_fia_rows_patched", False):
        return

    original_update = impl_cls.update_graph_params

    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        descriptor = getattr(forward_context, "batch_descriptor", None)
        runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
        use_live_fia_rows = bool(
            descriptor is not None
            and not getattr(descriptor, "uniform", True)
            and getattr(runtime_mode, "name", runtime_mode) == "FULL"
            and not using_paged_attention(num_tokens, vllm_config)
        )
        # The sink-FIA replay branch retains its captured block table, unlike
        # the ordinary FIA branch. Replace only that captured argument with the
        # stable runtime view whose row count matches actual_seq_lengths_q.
        if use_live_fia_rows and _EXTRA_CTX.sinks:
            graph_params = get_graph_params()
            captured_params = graph_params.attn_params[num_tokens]
            _dcut_rebind_live_fia_block_tables(
                captured_params,
                forward_context.attn_metadata,
            )
        return original_update(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            num_dcp_pcp_tokens,
            draft_attn_metadatas,
        )

    impl_cls.update_graph_params = staticmethod(update_graph_params)
    impl_cls._dcut_live_fia_rows_patched = True
    logger.info(
        "D-Cut: enabled live FIA request rows for ragged FULL graph replay."
    )


def _patch_graph_params_by_descriptor() -> None:
    """Make full-attention graph state safe for duplicate token buckets."""
    from vllm_ascend.compilation import acl_graph

    graph_params_cls = acl_graph.GraphParams
    if getattr(graph_params_cls, "_dcut_descriptor_patched", False):
        return

    original_init = graph_params_cls.__init__

    def __init__(self, events, workspaces, handles, attn_params):
        original_init(
            self,
            _DescriptorGraphParamMap(events, list_values=True),
            _DescriptorGraphParamMap(workspaces, list_values=False),
            _DescriptorGraphParamMap(handles, list_values=True),
            _DescriptorGraphParamMap(attn_params, list_values=True),
        )

    graph_params_cls.__init__ = __init__
    graph_params_cls._dcut_descriptor_patched = True
    logger.info(
        "D-Cut: isolated FULL graph attention parameters by BatchDescriptor."
    )


def _patch_full_decode_dispatcher() -> None:
    """Register ragged FULL keys beside stock uniform decode keys."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    dispatcher_cls = CudagraphDispatcher
    if getattr(dispatcher_cls, "_dcut_full_multishape_patched", False):
        return

    original_initialize = dispatcher_cls.initialize_cudagraph_keys

    def initialize_cudagraph_keys(
        self,
        cudagraph_mode,
        uniform_decode_query_len=1,
    ):
        original_initialize(
            self,
            cudagraph_mode,
            uniform_decode_query_len,
        )
        if (
            cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
            or not _dcut_full_decode_multishape_enabled(self.vllm_config)
        ):
            return

        # GraphParams are created after dispatcher initialization. Patch them
        # only for this mode so PIECEWISE keeps its original data structures.
        _patch_graph_params_by_descriptor()
        _patch_live_fia_graph_params()

        # Derive the token grid from the keys accepted by stock vLLM.  This
        # automatically preserves its speculative-query-length and max-seq
        # filtering instead of reimplementing configuration validation here.
        uniform_keys = tuple(
            descriptor
            for descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]
            if descriptor.uniform
        )
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        added = 0
        for descriptor in uniform_keys:
            ragged = self._create_padded_batch_descriptor(
                descriptor.num_tokens,
                False,
                descriptor.has_lora,
                descriptor.num_active_loras,
            )
            ragged = replace(
                ragged,
                uniform=False,
                num_reqs=_dcut_ragged_full_request_capacity(
                    descriptor.num_tokens,
                    max_num_seqs,
                ),
            )
            if ragged not in self.cudagraph_keys[CUDAGraphMode.FULL]:
                self.add_cudagraph_key(CUDAGraphMode.FULL, ragged)
                added += 1

        logger.warning(
            "D-Cut: registered %d ragged FULL_DECODE_ONLY graph keys "
            "beside %d uniform keys.",
            added,
            len(uniform_keys),
        )

    dispatcher_cls.initialize_cudagraph_keys = initialize_cudagraph_keys
    dispatcher_cls._dcut_full_multishape_patched = True


def _dcut_setup_full_decode_drafter(runner, drafter) -> None:
    """Keep the draft on stock uniform FULL graphs.

    The verifier descriptor becomes non-uniform after trimming, but the draft
    still emits exactly ``num_spec`` tokens per request.  Propagating the
    verifier's ragged descriptor would capture/replay an unnecessary and, for
    DFlash attention, unsupported non-uniform draft graph.
    """
    from vllm.config import CUDAGraphMode

    if not _dcut_full_decode_multishape_enabled(runner.vllm_config):
        return
    if (
        runner.compilation_config.cudagraph_mode
        != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        return
    if getattr(drafter, "_dcut_full_decode_multishape_setup", False):
        return

    if hasattr(drafter, "_propose"):
        original_propose = drafter._propose

        def _propose(*args, **kwargs):
            descriptor = kwargs.get("target_model_batch_desc")
            descriptor_arg = None
            if descriptor is None and len(args) > 6:
                descriptor = args[6]
                descriptor_arg = 6

            scheduler_output = kwargs.get("scheduler_output")
            if scheduler_output is None and len(args) > 13:
                scheduler_output = args[13]
            zero_draft_handoffs = getattr(
                runner,
                "_dcut_zero_draft_handoffs_for_proposal",
                frozenset(),
            )
            has_prefill = (
                _dcut_has_prefill(
                    runner,
                    scheduler_output,
                    zero_draft_handoffs,
                )
                if scheduler_output is not None
                else getattr(
                    runner,
                    "_dcut_gdn_scheduler_has_prefill",
                    None,
                )
            )
            if (
                descriptor is not None
                and not descriptor.uniform
                and has_prefill is False
            ):
                uniform_descriptor = replace(descriptor, uniform=True)
                if descriptor_arg is None:
                    kwargs = {
                        **kwargs,
                        "target_model_batch_desc": uniform_descriptor,
                    }
                else:
                    args = (
                        args[:descriptor_arg]
                        + (uniform_descriptor,)
                        + args[descriptor_arg + 1 :]
                    )

                # execute_model() and sample_tokens() are separate worker
                # calls. Carry the ragged-verifier origin explicitly while
                # the uniform draft prepares FIA metadata; the target-side
                # batch flag is not lifetime-safe here.
                padding_attr = "_dcut_drafter_from_nonuniform_decode"
                had_padding_attr = hasattr(runner, padding_attr)
                previous_padding = getattr(runner, padding_attr, None)
                setattr(runner, padding_attr, True)
                try:
                    return original_propose(*args, **kwargs)
                finally:
                    if had_padding_attr:
                        setattr(runner, padding_attr, previous_padding)
                    elif hasattr(runner, padding_attr):
                        delattr(runner, padding_attr)

            if descriptor is not None and not descriptor.uniform:
                # Prefill-containing target batches are eager. The ragged
                # FULL keys added for pure verifier decode must not make the
                # drafter graph such a batch after execute_model() returns.
                use_cuda_graph = getattr(drafter, "use_cuda_graph", None)
                if use_cuda_graph is not None:
                    drafter.use_cuda_graph = False
                    try:
                        return original_propose(*args, **kwargs)
                    finally:
                        drafter.use_cuda_graph = use_cuda_graph

            return original_propose(*args, **kwargs)

        drafter._propose = _propose


    drafter._dcut_full_decode_multishape_setup = True
    logger.warning(
        "D-Cut: FULL_DECODE_ONLY draft remains on uniform FULL graphs; "
        "ragged draft capture is disabled."
    )


def _dcut_validate_gdn_full_graph_weights(runner) -> None:
    """Verify every captured GDN layer belongs to the target model.

    ``qwen_gdn_attention_core`` resolves its layer through
    ``static_forward_context``.  A stale or draft-owned entry would make every
    multi-shape graph permanently capture the wrong ``conv1d/A_log/dt_bias``
    addresses.  Validate ownership after all weights are loaded and before the
    first graph capture; no weights are copied per shape.
    """
    from vllm.config import CUDAGraphMode

    if not _dcut_full_decode_multishape_enabled(runner.vllm_config):
        return
    if (
        runner.compilation_config.cudagraph_mode
        != CUDAGraphMode.FULL_DECODE_ONLY
        or not getattr(runner, "_has_gdn", False)
    ):
        return
    if getattr(runner, "pcp_size", 1) != 1 or getattr(runner, "dcp_size", 1) != 1:
        raise RuntimeError(
            "D-Cut ragged FULL_DECODE_ONLY GDN currently requires PCP=1 and DCP=1"
        )

    target_model = runner.get_model()
    target_module_ids = {id(module) for module in target_model.modules()}
    static_context = runner.compilation_config.static_forward_context
    bindings = []
    for prefix, layer in static_context.items():
        if not all(
            hasattr(layer, attr)
            for attr in ("conv1d", "A_log", "dt_bias", "_forward_core")
        ):
            continue
        if id(layer) not in target_module_ids:
            raise RuntimeError(
                "D-Cut FULL graph GDN binding is not owned by the target "
                f"model: prefix={prefix}, layer={type(layer).__name__}"
            )
        if getattr(layer, "prefix", prefix) != prefix:
            raise RuntimeError(
                "D-Cut FULL graph GDN prefix mismatch: "
                f"registry={prefix}, layer={getattr(layer, 'prefix', None)}"
            )

        tensors = (
            layer.conv1d.weight,
            layer.A_log,
            layer.dt_bias,
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise RuntimeError(
                f"D-Cut FULL graph GDN weights are not tensors: prefix={prefix}"
            )
        bindings.append(
            (prefix, *(int(tensor.data_ptr()) for tensor in tensors))
        )

    if not bindings:
        raise RuntimeError(
            "D-Cut detected GDN layers but found no target-owned GDN weight bindings"
        )
    runner._dcut_gdn_full_graph_weight_bindings = tuple(bindings)
    logger.warning(
        "D-Cut: validated %d target-owned GDN weight bindings for all "
        "FULL_DECODE_ONLY shapes (weights are shared, not copied).",
        len(bindings),
    )


def _patch_full_decode_multishape() -> None:
    """Install early, process-wide FULL_DECODE_ONLY support."""
    if not _dcut_full_decode_multishape_enabled():
        return
    _patch_full_decode_dispatcher()
