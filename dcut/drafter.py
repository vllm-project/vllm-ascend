# SPDX-License-Identifier: Apache-2.0
"""Patch live Ascend drafter instance for selected draft probs."""
from __future__ import annotations

from types import MethodType

import torch

from .globals import logger
from .utils import (
    _dcut_can_reuse_argmax_for_probs,
    _dcut_greedy_sample_with_selected_probs,
    _dcut_should_collect_draft_probs,
    _dcut_selected_token_probs,
)


def _dcut_attach_graph_owner(drafter) -> bool:
    """Bind a live proposer to its already-created ACLGraph wrapper."""
    runnable = getattr(drafter, "_runnable", None)
    runnable_state = getattr(runnable, "__dict__", None)
    if not (
        isinstance(runnable_state, dict)
        and "concrete_aclgraph_entries" in runnable_state
    ):
        return False
    runnable_state["_dcut_descriptor_owner"] = drafter
    drafter._dcut_graph_owner_attached = True
    return True


def _dcut_patch_drafter_instance(drafter) -> None:
    """Patch the live Ascend drafter instance; robust to MRO/load order quirks."""
    if _dcut_attach_graph_owner(drafter) and not getattr(
        drafter,
        "_dcut_logged_graph_owner_attached",
        False,
    ):
        logger.warning(
            "D-Cut: attached live drafter to its ACLGraphWrapper."
        )
        drafter._dcut_logged_graph_owner_attached = True
    if not hasattr(drafter, "take_last_selected_probs"):
        drafter.take_last_selected_probs = lambda: getattr(
            drafter, "_last_selected_probs", None
        )

    model = getattr(drafter, "model", None)
    if (
        model is not None
        and hasattr(model, "compute_logits")
        and not getattr(model, "_dcut_compute_logits_patched", False)
    ):
        orig_compute_logits = model.compute_logits

        def compute_logits(self_model, hidden_states, *args, **kwargs):
            logits = orig_compute_logits(hidden_states, *args, **kwargs)
            if _dcut_should_collect_draft_probs(drafter) and logits is not None:
                try:
                    can_reuse_argmax = _dcut_can_reuse_argmax_for_probs(
                        drafter
                    )
                    if can_reuse_argmax:
                        # _run_merged_draft returns the actual selected IDs.
                        # Keep logits alive so its shared eager/graph wrapper
                        # derives probabilities from exactly those IDs.
                        drafter._dcut_last_logits_for_probs = logits
                        drafter._last_selected_probs = None
                    else:
                        token_ids = logits.argmax(dim=-1)
                        selected_probs = _dcut_selected_token_probs(
                            logits,
                            token_ids,
                        )
                        drafter._last_selected_probs = (
                            selected_probs.float().contiguous()
                        )
                    if not getattr(
                        drafter, "_dcut_logged_compute_logits_probs", False
                    ):
                        logger.warning(
                            "D-Cut: captured selected draft probs/logits from "
                            "compute_logits on %s "
                            "(logits_shape=%s reuse_argmax=%s).",
                            type(drafter).__name__,
                            tuple(logits.shape),
                            can_reuse_argmax,
                        )
                        drafter._dcut_logged_compute_logits_probs = True
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "D-Cut: gather selected probs from compute_logits "
                        "failed: %s",
                        e,
                    )
                    drafter._last_selected_probs = None
            return logits

        model.compute_logits = MethodType(compute_logits, model)
        model._dcut_compute_logits_patched = True

    if (
        not hasattr(drafter, "compute_draft_token_ids")
        or getattr(drafter, "_dcut_instance_compute_patched", False)
    ):
        return

    orig_compute = drafter.compute_draft_token_ids

    def compute_draft_token_ids(self, hidden_states):
        self._last_selected_probs = None
        if not _dcut_should_collect_draft_probs(self):
            return orig_compute(hidden_states)
        try:
            logits = self.model.logits_processor(self.model.lm_head, hidden_states)
            logits = logits.contiguous()
            next_token, selected_probs = _dcut_greedy_sample_with_selected_probs(
                logits
            )
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
                "D-Cut: gather selected probs in live drafter failed: %s", e
            )
            self._last_selected_probs = None
            return orig_compute(hidden_states)

    drafter.compute_draft_token_ids = MethodType(compute_draft_token_ids, drafter)
    drafter._dcut_instance_compute_patched = True
