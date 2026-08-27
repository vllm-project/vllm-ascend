# SPDX-License-Identifier: Apache-2.0
"""Install the graph-capturable D-Cut GDN core for vLLM 0.23."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from .globals import logger

ENV_TORCH_OP_LIBRARY = "VLLM_DCUT_TORCH_OP_LIBRARY"
_REQUIRED_OPS = (
    "npu_dcut_causal_conv1d",
    "npu_dcut_recurrent_gated_delta_rule",
)


def _dcut_gdn_has_prefill(forward_context, prefix: str | None = None) -> bool:
    """Return whether the current GDN batch contains prefill work."""
    if forward_context is None:
        return False
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return False
    metadata = (
        attn_metadata.values()
        if prefix is None
        else (attn_metadata.get(prefix),)
    )
    return any(
        int(getattr(meta, "num_prefills", 0)) > 0
        for meta in metadata
        if meta is not None
    )


def _dcut_gdn_use_native_core(forward_context, prefix: str) -> bool:
    """Route prefill-containing batches around every D-Cut GDN operator."""
    if forward_context is None:
        return False

    # D-Cut adds non-uniform FULL descriptors next to vLLM's stock uniform
    # descriptors. Keep the stock descriptors bit-for-bit native, including
    # recompute handoff; only the added ragged descriptors need D-Cut's core.
    runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    descriptor = getattr(forward_context, "batch_descriptor", None)
    if (
        getattr(runtime_mode, "name", runtime_mode) == "FULL"
        and bool(getattr(descriptor, "uniform", False))
    ):
        return True

    # ``GDNAttentionMetadata.num_prefills`` is overloaded for mixed
    # speculative/non-speculative decode: the native builder folds ordinary
    # decode rows into that count even though no prompt tokens are present.
    # When the model runner has propagated the scheduler's real-prefill
    # decision, it must therefore be authoritative, including when False.
    native_batch = getattr(
        forward_context,
        "_dcut_gdn_native_batch",
        None,
    )
    if native_batch is not None:
        return bool(native_batch)

    # Dummy graph capture and direct unit paths do not have a SchedulerOutput.
    return _dcut_gdn_has_prefill(forward_context, prefix)


def _ops_registered() -> bool:
    return all(hasattr(torch.ops._C_ascend, name) for name in _REQUIRED_OPS)


def _load_dcut_torch_ops() -> bool:
    """Load the D-Cut-only Torch registration library before graph capture."""
    if _ops_registered():
        return True

    configured_path = os.environ.get(ENV_TORCH_OP_LIBRARY)
    if configured_path:
        candidates = (Path(configured_path).expanduser(),)
    else:
        candidates = (
            Path(__file__).resolve().parent
            / "kernel"
            / "build"
            / "torch_extension"
            / "dcut_torch_ops.so",
        )

    load_errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            load_errors.append(f"{candidate} (not found)")
            continue
        try:
            torch.ops.load_library(str(candidate))
        except (OSError, RuntimeError) as exc:
            load_errors.append(f"{candidate} ({exc})")
            continue
        if _ops_registered():
            return True
        load_errors.append(f"{candidate} (loaded, but schemas are missing)")

    logger.error(
        "D-Cut: custom GDN Torch operators are unavailable: %s. "
        "Build them with `bash dcut/kernel/build.sh` or set %s.",
        "; ".join(load_errors),
        ENV_TORCH_OP_LIBRARY,
    )
    return False


def _patch_gdn_spec_metadata_builder(gdn_attn_builder) -> None:
    """Use compact D-Cut metadata only for decode-only batches."""
    target_class = gdn_attn_builder.AscendGDNAttentionMetadataBuilder
    current_build = target_class.build
    if not getattr(current_build, "_dcut_fia_dummy_patched", False):

        def _dcut_build(
            self,
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=None,
            num_decode_draft_tokens_cpu=None,
            fast_build=False,
        ):
            configured_mode = getattr(
                self.vllm_config.compilation_config.cudagraph_mode,
                "name",
                None,
            )
            if (
                configured_mode != "FULL_DECODE_ONLY"
                or getattr(
                    self,
                    "_dcut_force_native_gdn_metadata",
                    False,
                )
            ):
                return current_build(
                    self,
                    common_prefix_len,
                    common_attn_metadata,
                    num_accepted_tokens,
                    num_decode_draft_tokens_cpu,
                    fast_build,
                )

            # FULL attention appends a virtual FIA request to reach padded Q,
            # while GDN intentionally receives its unpadded query_start_loc.
            # Keep that FIA-only row out of GDN classification/state updates.
            query_start_loc_cpu = (
                common_attn_metadata.query_start_loc_cpu
            )
            actual_num_reqs = min(
                common_attn_metadata.num_reqs,
                len(query_start_loc_cpu) - 1,
            )
            # GDN qsl keeps real boundaries, so FIA padding appears as one
            # or more trailing zero-length rows. Real scheduled requests always
            # own at least the verifier backbone token. This is the pinned CPU
            # snapshot, so the scalar checks below never synchronize the NPU.
            while (
                actual_num_reqs > 0
                and int(query_start_loc_cpu[actual_num_reqs])
                == int(query_start_loc_cpu[actual_num_reqs - 1])
            ):
                actual_num_reqs -= 1
            if actual_num_reqs < common_attn_metadata.num_reqs:
                common_attn_metadata = common_attn_metadata.unpadded(
                    common_attn_metadata.num_actual_tokens,
                    actual_num_reqs,
                )
                if num_accepted_tokens is not None:
                    num_accepted_tokens = num_accepted_tokens[:actual_num_reqs]
                if num_decode_draft_tokens_cpu is not None:
                    num_decode_draft_tokens_cpu = (
                        num_decode_draft_tokens_cpu[:actual_num_reqs]
                    )
            return current_build(
                self,
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens,
                num_decode_draft_tokens_cpu,
                fast_build,
            )

        _dcut_build._dcut_fia_dummy_patched = True
        target_class.build = _dcut_build

    current = target_class._attach_spec_decode_metadata
    if getattr(current, "_dcut_patched", False):
        return

    def _dcut_attach_spec_decode_metadata(self, attn_metadata):
        # Prefill-containing batches bypass the D-Cut GDN core. Preserve the
        # native actual_seq_lengths layout consumed by its recurrent kernel.
        if (
            int(attn_metadata.num_prefills) > 0
            or getattr(
                self,
                "_dcut_force_native_gdn_metadata",
                False,
            )
        ):
            return current(self, attn_metadata)

        attn_metadata.spec_decode_metadata = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError(
                "Expected attn_metadata.spec_query_start_loc for Ascend "
                "GDN speculative path."
            )
        if attn_metadata.spec_state_indices_tensor is None:
            raise RuntimeError(
                "Expected spec_state_indices_tensor for Ascend GDN "
                "speculative conv1d path."
            )
        if attn_metadata.num_accepted_tokens is None:
            raise RuntimeError(
                "Expected num_accepted_tokens for Ascend GDN speculative "
                "conv1d path."
            )

        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1
        attn_metadata.spec_decode_metadata = (
            gdn_attn_builder.GDNSpecDecodeMetadata(
                spec_causal_conv1d=gdn_attn_builder.GDNSpecCausalConv1dMetadata(
                    query_start_loc=attn_metadata.spec_query_start_loc,
                    cache_indices=attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                    num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
                ),
                actual_seq_lengths=attn_metadata.spec_query_start_loc,
            )
        )
        return attn_metadata

    _dcut_attach_spec_decode_metadata._dcut_patched = True
    target_class._attach_spec_decode_metadata = _dcut_attach_spec_decode_metadata


def _patch_gdn_dcut() -> bool:
    """Patch GDN routing and expose the graphable pure-spec PIECEWISE path."""
    try:
        from vllm.forward_context import get_forward_context
        from vllm_ascend.ops import gdn as ascend_gdn
        from vllm_ascend.ops import gdn_attn_builder
        from vllm_ascend.patch.worker import patch_qwen3_5 as qwen_patch
        from vllm_ascend.utils import is_310p
    except Exception as exc:  # pragma: no cover - depends on runtime imports
        logger.warning("D-Cut: cannot import vLLM 0.23 GDN symbols: %s", exc)
        return False

    if is_310p():
        logger.warning("D-Cut: variable-length GDN verification is not enabled on 310P.")
        return False

    target_class = qwen_patch._GDN_PATCH_TARGET

    if not _load_dcut_torch_ops():
        return False

    _patch_gdn_spec_metadata_builder(gdn_attn_builder)
    if (
        getattr(target_class._forward_core, "_dcut_patched", False)
        and getattr(target_class.forward, "_dcut_patched", False)
    ):
        return True

    from .gdn_forward_v023 import AscendGatedDeltaNetAttention as DcutGatedDeltaNetAttention

    native_forward_core = target_class._forward_core
    native_forward = target_class.forward
    dcut_forward_core = DcutGatedDeltaNetAttention._forward_core
    graphable_spec_forward = (
        DcutGatedDeltaNetAttention.forward_with_graphable_recurrent
    )

    def _dcut_forward_core(
        self,
        mixed_qkv,
        b,
        a,
        core_attn_out,
    ):
        forward_context = get_forward_context()
        if _dcut_gdn_use_native_core(forward_context, self.prefix):
            return native_forward_core(
                self,
                mixed_qkv,
                b,
                a,
                core_attn_out,
            )
        return dcut_forward_core(
            self,
            mixed_qkv,
            b,
            a,
            core_attn_out,
        )

    _dcut_forward_core._dcut_patched = True  # type: ignore[attr-defined]
    _dcut_forward_core._dcut_native_forward_core = (  # type: ignore[attr-defined]
        native_forward_core
    )

    def _dcut_forward(self, hidden_states, output):
        forward_context = get_forward_context()
        piecewise_graph_safe = getattr(
            forward_context,
            "_dcut_gdn_recurrent_piecewise_safe",
            False,
        )
        full_graph_safe = getattr(
            forward_context,
            "_dcut_gdn_full_graph_safe",
            False,
        )
        if piecewise_graph_safe or full_graph_safe:
            # Use the same expanded GDN path in both graph modes. In
            # particular, ragged FULL must not re-enter the opaque
            # qwen_gdn_attention_core custom op: its hidden recurrent-state
            # mutation is not part of that op's explicit tensor signature.
            # The graphable recurrent wrapper declares ``state`` as mutated
            # and is already accuracy-validated by the PIECEWISE path.
            return graphable_spec_forward(
                self,
                hidden_states,
                output,
            )
        return native_forward(self, hidden_states, output)

    _dcut_forward._dcut_patched = True  # type: ignore[attr-defined]
    _dcut_forward._dcut_native_forward = native_forward  # type: ignore[attr-defined]

    ascend_gdn.AscendGatedDeltaNetAttention._forward_core = _dcut_forward_core
    target_class._forward_core = _dcut_forward_core
    ascend_gdn.AscendGatedDeltaNetAttention.forward = _dcut_forward
    target_class.forward = _dcut_forward
    logger.info(
        "D-Cut: enabled native prefill/mixed GDN routing and the "
        "graphable pure-spec PIECEWISE/FULL core."
    )
    return True
