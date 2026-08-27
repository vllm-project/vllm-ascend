# SPDX-License-Identifier: Apache-2.0
"""Patch Ascend spec-decode proposer for selected draft probs."""
from __future__ import annotations

import os

import torch
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from .globals import ENV_CONFIG, logger
from .utils import (
    _dcut_greedy_sample_with_selected_probs,
    _dcut_in_graph_capture,
    _dcut_should_collect_draft_probs,
    _dcut_selected_token_probs,
)

# ---------------------------------------------------------------------------
# Patch installers (idempotent, per class).  Targets are the *NPU* classes.
# ---------------------------------------------------------------------------


def _dcut_store_unique_graph_buffer(owner, attr_name, key, tensor) -> None:
    """Retain *tensor* only while *key* identifies one graph allocation.

    Ascend uses a global ACLGraph pool, so different graph entries are allowed
    to reuse an output address. Shape and numel keys are even less specific.
    Mark a key ambiguous instead of silently replacing it with the final graph
    captured during startup.
    """
    buffers = getattr(owner, attr_name, None)
    if buffers is None:
        buffers = {}
        setattr(owner, attr_name, buffers)
    if key not in buffers:
        buffers[key] = tensor
        return
    existing = buffers[key]
    if existing is None:
        return
    if int(existing.data_ptr()) != int(tensor.data_ptr()):
        buffers[key] = None


def _dcut_selected_probs_for_output(logits, output):
    """Gather probabilities for the exact draft IDs returned to vLLM."""
    flat_token_ids = output.reshape(-1)
    n_tokens = min(int(logits.shape[0]), int(flat_token_ids.numel()))
    selected_probs = _dcut_selected_token_probs(
        logits[:n_tokens],
        flat_token_ids[:n_tokens],
    )
    if selected_probs.numel() == output.numel():
        selected_probs = selected_probs.view(output.shape)
    return selected_probs.float().contiguous()


def _dcut_register_graph_selected_probs(
    owner,
    descriptor,
    output,
    selected_probs,
) -> bool:
    """Register the probability tensor retained by one captured draft graph."""
    if selected_probs is None:
        return False

    if descriptor is not None:
        by_descriptor = getattr(
            owner,
            "_dcut_graph_selected_probs_by_descriptor",
            None,
        )
        if by_descriptor is None:
            by_descriptor = {}
            owner._dcut_graph_selected_probs_by_descriptor = by_descriptor
        by_descriptor[descriptor] = selected_probs

    _dcut_store_unique_graph_buffer(
        owner,
        "_dcut_graph_selected_probs_by_output_ptr",
        int(output.data_ptr()),
        selected_probs,
    )
    _dcut_store_unique_graph_buffer(
        owner,
        "_dcut_graph_selected_probs_by_shape",
        tuple(output.shape),
        selected_probs,
    )
    _dcut_store_unique_graph_buffer(
        owner,
        "_dcut_graph_selected_probs_by_numel",
        int(output.numel()),
        selected_probs,
    )
    owner._dcut_graph_selected_probs_ready = True
    return True


def _dcut_graph_owner_from_runnable(runnable):
    """Return the draft proposer owning an ACLGraph bound runnable."""
    owner = getattr(runnable, "__self__", None)
    if owner is None:
        return None
    if not hasattr(owner, "compute_draft_token_ids"):
        return None
    if not hasattr(owner, "_run_merged_draft"):
        return None
    return owner


def _dcut_prepare_dflash_graph_context(
    owner,
    descriptor,
    graph_context_tokens,
    capture_pending: bool,
):
    """Mask the fixed-shape DFlash context tail before capture/replay."""
    if getattr(owner, "method", None) != "dflash" or descriptor is None:
        return None

    slot_mapping = getattr(owner, "_context_slot_mapping_buffer", None)
    actual_context = getattr(owner, "_dflash_num_context", None)
    if slot_mapping is None or actual_context is None:
        return None

    actual_context = int(actual_context)
    captured_by_descriptor = getattr(
        owner,
        "_dcut_dflash_context_tokens_by_descriptor",
        None,
    )
    if captured_by_descriptor is None:
        captured_by_descriptor = {}
        owner._dcut_dflash_context_tokens_by_descriptor = (
            captured_by_descriptor
        )

    restore_context = None
    if capture_pending:
        if graph_context_tokens is None:
            return None
        captured_context = min(
            int(graph_context_tokens),
            int(slot_mapping.shape[0]),
        )
        captured_by_descriptor[descriptor] = captured_context
        restore_context = actual_context
    else:
        captured_context = captured_by_descriptor.get(descriptor)
        if captured_context is None:
            return None

    if actual_context < 0 or actual_context > captured_context:
        raise RuntimeError(
            "D-Cut DFlash graph context exceeds its captured length: "
            f"actual={actual_context}, captured={captured_context}, "
            f"descriptor={descriptor}."
        )

    if capture_pending:
        owner._dflash_num_context = captured_context

    tail_masked = actual_context < captured_context
    if tail_masked:
        # Keep stale graph-tail slots out of the replayed DFlash context.
        slot_mapping[actual_context:captured_context].fill_(
            PADDING_SLOT_ID
        )
        if not getattr(
            owner,
            "_dcut_logged_dflash_context_padding",
            False,
        ):
            logger.warning(
                "D-Cut: masked the DFlash FULL-graph context tail with "
                "a standalone fill (actual=%d captured=%d descriptor=%s).",
                actual_context,
                captured_context,
                descriptor,
            )
            owner._dcut_logged_dflash_context_padding = True

    owner._dcut_last_dflash_context_actual = actual_context
    owner._dcut_last_dflash_context_captured = captured_context
    owner._dcut_last_dflash_context_tail_masked = tail_masked
    return restore_context


def _patch_aclgraph_descriptor_tracking() -> None:
    """Expose the exact draft ``BatchDescriptor`` selected for replay."""
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    if getattr(ACLGraphWrapper, "_dcut_descriptor_tracking_patched", False):
        return

    original_init = ACLGraphWrapper.__init__
    original_call = ACLGraphWrapper.__call__

    def __init__(graph_wrapper, runnable, *args, **kwargs):
        original_init(graph_wrapper, runnable, *args, **kwargs)
        proposer = _dcut_graph_owner_from_runnable(runnable)
        if proposer is not None:
            graph_wrapper.__dict__["_dcut_descriptor_owner"] = proposer
            proposer._dcut_graph_owner_attached = True

    def __call__(graph_wrapper, *args, **kwargs):
        forward_context = get_forward_context()
        descriptor = getattr(forward_context, "batch_descriptor", None)
        proposer = graph_wrapper.__dict__.get("_dcut_descriptor_owner")
        if proposer is None:
            runnable = graph_wrapper.__dict__.get("runnable")
            proposer = _dcut_graph_owner_from_runnable(runnable)
            if proposer is not None:
                graph_wrapper.__dict__["_dcut_descriptor_owner"] = proposer
                proposer._dcut_graph_owner_attached = True
        capture_pending = False
        graph_active = False
        if proposer is not None:
            proposer._dcut_current_graph_descriptor = descriptor
            runtime_mode = getattr(
                forward_context,
                "cudagraph_runtime_mode",
                None,
            )
            entries = getattr(graph_wrapper, "concrete_aclgraph_entries", {})
            entry = entries.get(descriptor)
            graph_active = runtime_mode == getattr(
                graph_wrapper,
                "runtime_mode",
                None,
            )
            capture_pending = graph_active and (
                entry is None or getattr(entry, "aclgraph", None) is None
            )

        restore_dflash_context = None
        if proposer is not None and graph_active:
            restore_dflash_context = _dcut_prepare_dflash_graph_context(
                proposer,
                descriptor,
                kwargs.get("num_input_tokens"),
                capture_pending,
            )
        try:
            output = original_call(graph_wrapper, *args, **kwargs)
        finally:
            if restore_dflash_context is not None:
                proposer._dflash_num_context = restore_dflash_context
        if proposer is None or not capture_pending:
            return output

        selected_probs = getattr(proposer, "_last_selected_probs", None)
        if _dcut_register_graph_selected_probs(
            proposer,
            descriptor,
            output,
            selected_probs,
        ):
            if not getattr(
                proposer,
                "_dcut_logged_graph_prob_capture",
                False,
            ):
                logger.warning(
                    "D-Cut: registered output-aligned draft probs directly after "
                    "ACLGraph capture (descriptor=%s output_shape=%s "
                    "probs_shape=%s).",
                    descriptor,
                    tuple(output.shape),
                    tuple(selected_probs.shape),
                )
                proposer._dcut_logged_graph_prob_capture = True
        elif not getattr(
            proposer,
            "_dcut_logged_missing_graph_prob_capture",
            False,
        ):
            logger.warning(
                "D-Cut: ACLGraph capture completed without a selected-prob "
                "tensor (descriptor=%s needs=%s method=%s).",
                descriptor,
                getattr(proposer, "needs_draft_probs", False),
                getattr(proposer, "method", None),
            )
            proposer._dcut_logged_missing_graph_prob_capture = True
        return output

    ACLGraphWrapper.__init__ = __init__
    ACLGraphWrapper.__call__ = __call__
    ACLGraphWrapper._dcut_descriptor_tracking_patched = True


def _patch_proposer() -> None:
    """Patch the Ascend spec-decode proposer to expose selected draft probs.

    vLLM 0.23 Ascend proposers sample through compute_draft_token_ids.
    Patch the concrete method owner so DFlash and parallel draft-model
    proposers expose the selected-token probabilities used by D-Cut.
    """
    from vllm.forward_context import get_forward_context
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    _patch_aclgraph_descriptor_tracking()

    # Collect the concrete Ascend proposers that can run D-Cut (dflash / PARD).
    proposer_classes = []
    try:
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
        proposer_classes.append(AscendDflashProposer)
    except Exception as e:  # pragma: no cover
        logger.warning("D-Cut: could not import AscendDflashProposer: %s", e)
    try:
        from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
        proposer_classes.append(AscendDraftModelProposer)
    except Exception:  # PARD path is optional
        pass

    # Helper functions live on the shared base so every proposer can call them.
    if not getattr(SpecDecodeBaseProposer, "_dcut_helpers", False):
        @staticmethod
        def _should_collect_draft_probs(self):
            return _dcut_should_collect_draft_probs(self)

        @staticmethod
        def _gather_selected_probs(logits, token_ids, full_probs):
            idx = token_ids.long().unsqueeze(-1)
            if full_probs is not None:
                return full_probs.gather(-1, idx).squeeze(-1)
            return _dcut_selected_token_probs(logits, token_ids)

        @staticmethod
        def _greedy_sample_with_selected_probs(logits):
            return _dcut_greedy_sample_with_selected_probs(logits)

        def take_last_selected_probs(self):
            return getattr(self, "_last_selected_probs", None)

        SpecDecodeBaseProposer.needs_draft_probs = bool(os.environ.get(ENV_CONFIG))
        SpecDecodeBaseProposer._last_selected_probs = None
        SpecDecodeBaseProposer._should_collect_draft_probs = (
            _should_collect_draft_probs
        )
        SpecDecodeBaseProposer._gather_selected_probs = _gather_selected_probs
        SpecDecodeBaseProposer._greedy_sample_with_selected_probs = (
            _greedy_sample_with_selected_probs
        )
        SpecDecodeBaseProposer.take_last_selected_probs = take_last_selected_probs
        SpecDecodeBaseProposer._dcut_helpers = True

    compute_owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "compute_draft_token_ids" in klass.__dict__:
                if klass not in compute_owners:
                    compute_owners.append(klass)
                break

    for owner in compute_owners:
        if getattr(owner, "_dcut_compute_patched", False):
            continue
        _orig_compute = owner.compute_draft_token_ids

        def _make_compute_wrapper(orig):
            def compute_draft_token_ids(self, hidden_states):
                self._last_selected_probs = None
                if not type(self)._should_collect_draft_probs(self):
                    return orig(self, hidden_states)
                try:
                    logits = self.model.logits_processor(
                        self.model.lm_head, hidden_states
                    )
                    logits = logits.contiguous()
                    next_token, selected_probs = (
                        type(self)._greedy_sample_with_selected_probs(logits)
                    )
                    # Keep this flat here. Ascend may pad sample_hidden_states for
                    # lmhead TP; the runner slices and reshapes using real batch size.
                    self._last_selected_probs = selected_probs.float().contiguous()

                    draft_map = getattr(self.model, "draft_id_to_target_id", None)
                    if draft_map is None:
                        return next_token
                    bias = torch.index_select(
                        draft_map, dim=0, index=next_token.view(-1)
                    ).view(next_token.shape)
                    return next_token + bias
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: gather selected probs in compute_draft_token_ids "
                        "failed: %s",
                        e,
                    )
                    self._last_selected_probs = None
                    return orig(self, hidden_states)

            return compute_draft_token_ids

        owner.compute_draft_token_ids = _make_compute_wrapper(_orig_compute)
        owner._dcut_compute_patched = True
        logger.info(
            "D-Cut: patched compute_draft_token_ids on %s.", owner.__name__
        )

    run_merged_owners = []
    for pc in proposer_classes:
        for klass in pc.__mro__:
            if "_run_merged_draft" in klass.__dict__:
                if klass not in run_merged_owners:
                    run_merged_owners.append(klass)
                break

    for owner in run_merged_owners:
        if getattr(owner, "_dcut_run_merged_patched", False):
            continue
        original_run_merged = owner._run_merged_draft

        def _make_run_merged_wrapper(orig):
            def _run_merged_draft(self, *args, **kwargs):
                in_graph_capture = _dcut_in_graph_capture()
                self._dcut_last_draft_ran_python = True
                self._dcut_last_logits_for_probs = None
                self._last_selected_probs = None
                out = orig(self, *args, **kwargs)
                if not type(self)._should_collect_draft_probs(self):
                    return out

                selected_probs = getattr(self, "_last_selected_probs", None)
                logits = getattr(self, "_dcut_last_logits_for_probs", None)
                if selected_probs is None and logits is not None:
                    try:
                        # Use the IDs actually returned by the proposer. This
                        # is the PIECEWISE behavior and keeps FULL aligned when
                        # DFlash applies vocab remapping, padding, or tie breaks.
                        selected_probs = _dcut_selected_probs_for_output(
                            logits,
                            out,
                        )
                        self._last_selected_probs = selected_probs
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(
                            "D-Cut: output-aligned selected-prob capture "
                            "failed: %s",
                            e,
                        )
                        self._last_selected_probs = None
                        selected_probs = None
                    finally:
                        self._dcut_last_logits_for_probs = None
                        logits = None
                if in_graph_capture:
                    # Bind the probability tensor to the exact graph descriptor.
                    # Replay updates these tensors without running this Python
                    # wrapper again. Output address, shape and numel are retained
                    # only as compatibility fallbacks when they are unique.
                    if selected_probs is not None:
                        forward_context = get_forward_context()
                        self._dcut_current_graph_descriptor = getattr(
                            forward_context,
                            "batch_descriptor",
                            None,
                        )
                        runnable = getattr(self, "_runnable", None)
                        runnable_state = getattr(runnable, "__dict__", None)
                        if (
                            isinstance(runnable_state, dict)
                            and "concrete_aclgraph_entries" in runnable_state
                        ):
                            runnable_state["_dcut_descriptor_owner"] = self
                        descriptor = getattr(
                            self,
                            "_dcut_current_graph_descriptor",
                            None,
                        )
                        if descriptor is not None:
                            by_descriptor = getattr(
                                self,
                                "_dcut_graph_selected_probs_by_descriptor",
                                None,
                            )
                            if by_descriptor is None:
                                by_descriptor = {}
                                self._dcut_graph_selected_probs_by_descriptor = (
                                    by_descriptor
                                )
                            by_descriptor[descriptor] = selected_probs
                        _dcut_store_unique_graph_buffer(
                            self,
                            "_dcut_graph_selected_probs_by_output_ptr",
                            int(out.data_ptr()),
                            selected_probs,
                        )
                        _dcut_store_unique_graph_buffer(
                            self,
                            "_dcut_graph_selected_probs_by_shape",
                            tuple(out.shape),
                            selected_probs,
                        )
                        _dcut_store_unique_graph_buffer(
                            self,
                            "_dcut_graph_selected_probs_by_numel",
                            int(out.numel()),
                            selected_probs,
                        )
                        self._dcut_graph_selected_probs_ready = True
                        if not getattr(
                            self,
                            "_dcut_logged_graph_prob_capture",
                            False,
                        ):
                            logger.warning(
                                "D-Cut: captured output-aligned draft probs in ACLGraph "
                                "(descriptor=%s output_shape=%s probs_shape=%s).",
                                descriptor,
                                tuple(out.shape),
                                tuple(selected_probs.shape),
                            )
                            self._dcut_logged_graph_prob_capture = True

                    # Retain logits only as a compatibility fallback for a
                    # proposer path that could not produce probabilities while
                    # being captured.
                    if logits is not None:
                        descriptor = getattr(
                            self,
                            "_dcut_current_graph_descriptor",
                            None,
                        )
                        if descriptor is not None:
                            by_descriptor = getattr(
                                self,
                                "_dcut_graph_logits_for_probs_by_descriptor",
                                None,
                            )
                            if by_descriptor is None:
                                by_descriptor = {}
                                self._dcut_graph_logits_for_probs_by_descriptor = (
                                    by_descriptor
                                )
                            by_descriptor[descriptor] = logits
                        _dcut_store_unique_graph_buffer(
                            self,
                            "_dcut_graph_logits_for_probs_by_shape",
                            tuple(out.shape),
                            logits,
                        )
                        _dcut_store_unique_graph_buffer(
                            self,
                            "_dcut_graph_logits_for_probs_by_numel",
                            int(out.numel()),
                            logits,
                        )
                        self._dcut_graph_logits_for_probs_ready = True
                    if (
                        selected_probs is None
                        and not getattr(
                            self,
                            "_dcut_logged_missing_graph_prob_capture",
                            False,
                        )
                    ):
                        logger.warning(
                            "D-Cut: ACLGraph capture produced no selected draft "
                            "probs (needs=%s method=%s parallel=%s logits=%s "
                            "descriptor=%s).",
                            getattr(self, "needs_draft_probs", False),
                            getattr(self, "method", None),
                            getattr(self, "parallel_drafting", False),
                            logits is not None,
                            getattr(
                                self,
                                "_dcut_current_graph_descriptor",
                                None,
                            ),
                        )
                        self._dcut_logged_missing_graph_prob_capture = True
                    return out

                return out

            return _run_merged_draft

        owner._run_merged_draft = _make_run_merged_wrapper(
            original_run_merged
        )
        owner._dcut_run_merged_patched = True
        logger.info(
            "D-Cut: patched _run_merged_draft on %s.",
            owner.__name__,
        )
