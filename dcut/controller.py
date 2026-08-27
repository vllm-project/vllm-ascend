# SPDX-License-Identifier: Apache-2.0
"""Controller init + drafter-probs enablement."""
from __future__ import annotations

import os

import torch

from .globals import (
    ENV_CONFIG,
    ENV_DISABLE,
    ENV_FORCE_DRAFTER_EAGER,
    ENV_FULL_DECODE_ONLY,
    ENV_SKIP_UNREADY_PROBS,
    ENV_TRIM_STATS_OUT,
    logger,
)
from .utils import (
    _dcut_process_probs_stage,
    _dcut_reuse_argmax_enabled,
    _env_flag,
    _npu_event,
    _supports_adaptive_verify,
)
from .drafter import _dcut_patch_drafter_instance
from .patch_full_graph import _dcut_setup_full_decode_drafter
from .verify_adaptive_config import VerifyAdaptiveConfig
from .verify_adaptive_controller import VerifyAdaptiveController

# ---------------------------------------------------------------------------
# Runner-side helpers (installed as methods or used by the wrappers).
# Device-agnostic except where noted; identical to the CUDA plugin.
# ---------------------------------------------------------------------------

def _dcut_init_controller(self) -> None:
    """Build the controller + async-probs buffers on an NPUModelRunner instance.

    Enabled iff ``VLLM_DCUT_CONFIG`` points to a JSON config AND the speculative
    method is parallel (dflash / PARD).  Otherwise leaves the runner untouched.
    """
    self._verify_adaptive_controller = None
    self._adaptive_probs_event = None
    self._adaptive_probs_pinned = None
    self._adaptive_probs_pending = False
    self._adaptive_probs_expired = False
    self._adaptive_probs_source = "none"
    self._adaptive_probs_last_consumed_source = "none"
    self._adaptive_probs_generation = 0
    self._adaptive_probs_last_consumed_generation = 0
    self._adaptive_probs_last_consumed_mean_by_position = []
    self._adaptive_num_reqs = 0
    self._adaptive_req_ids = []
    self._adaptive_active = set()
    # Verify-reduction stats (how much D-Cut trimmed); logged every N steps.
    self._dcut_stat_full = 0
    self._dcut_stat_trimmed = 0
    self._dcut_stat_reqs = 0
    self._dcut_stat_steps = 0
    self._dcut_stat_log_every = int(os.environ.get("VLLM_DCUT_STAT_EVERY", "200") or 0)
    self._dcut_trim_stats_out = os.environ.get(ENV_TRIM_STATS_OUT) or None
    self._dcut_skip_unready_probs = _env_flag(ENV_SKIP_UNREADY_PROBS)
    self._dcut_process_probs_stage = _dcut_process_probs_stage()
    self._dcut_missing_probs_steps = 0
    self._dcut_logged_drafter_probs = False

    cfg_path = os.environ.get(ENV_CONFIG) or None
    if not cfg_path:
        return

    if _env_flag(ENV_DISABLE):
        logger.info(
            "D-Cut trimming and draft-probability control disabled by %s; "
            "D-Cut GDN operator patches remain active.",
            ENV_DISABLE,
        )
        return

    if os.environ.get(ENV_FULL_DECODE_ONLY):
        logger.info(
            "D-Cut adaptive verify disabled by %s; running full decode-only "
            "baseline with the D-Cut plugin loaded.",
            ENV_FULL_DECODE_ONLY,
        )
        return

    spec_cfg = getattr(self, "speculative_config", None)
    if not _supports_adaptive_verify(spec_cfg):
        logger.warning(
            "VLLM_DCUT_CONFIG is set but the speculative method does not support "
            "adaptive verifier step-length (requires dflash or "
            "draft_model+parallel_drafting); D-Cut disabled."
        )
        return

    num_spec = getattr(self, "num_spec_tokens", 0) or 0
    if num_spec <= 0:
        logger.warning("D-Cut: num_spec_tokens <= 0; disabled.")
        return

    acfg = VerifyAdaptiveConfig.from_json(cfg_path)
    self._verify_adaptive_controller = VerifyAdaptiveController(
        config=acfg,
        num_spec_tokens=num_spec,
        max_batch_size=self.scheduler_config.max_num_seqs,
        device=self.device,
    )
    # NPU: torch.npu.Event instead of torch.cuda.Event.
    self._adaptive_probs_event = _npu_event()
    self._adaptive_probs_pinned = torch.empty(
        (self.max_num_reqs, num_spec),
        dtype=torch.float32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    _dcut_enable_drafter_probs(self)
    logger.info(
        "D-Cut adaptive verify ENABLED on NPU "
        "(config=%s process_probs_stage=%s skip_unready_probs=%s "
        "reuse_argmax=%s).",
        cfg_path,
        self._dcut_process_probs_stage,
        self._dcut_skip_unready_probs,
        _dcut_reuse_argmax_enabled(),
    )


def _dcut_enable_drafter_probs(self) -> None:
    """Enable draft-prob collection once the Ascend drafter object exists."""
    if getattr(self, "_verify_adaptive_controller", None) is None:
        return
    drafter = getattr(self, "drafter", None)
    if drafter is None:
        return
    if not hasattr(drafter, "needs_draft_probs"):
        if not getattr(self, "_dcut_logged_drafter_probs", False):
            logger.warning(
                "D-Cut: drafter %s has no needs_draft_probs flag.",
                type(drafter).__name__,
            )
            self._dcut_logged_drafter_probs = True
        return
    _dcut_patch_drafter_instance(drafter)
    _dcut_setup_full_decode_drafter(self, drafter)
    if _env_flag(ENV_FORCE_DRAFTER_EAGER):
        drafter.use_cuda_graph = False
        if not getattr(self, "_dcut_logged_force_drafter_eager", False):
            logger.warning(
                "D-Cut: forced drafter eager via %s; target graph mode "
                "remains %s.",
                ENV_FORCE_DRAFTER_EAGER,
                getattr(
                    getattr(self, "compilation_config", None),
                    "cudagraph_mode",
                    None,
                ),
            )
            self._dcut_logged_force_drafter_eager = True
    drafter.needs_draft_probs = True
    if not getattr(self, "_dcut_logged_drafter_probs", False):
        logger.warning(
            "D-Cut: enabled selected draft probs on drafter %s "
            "(method=%s parallel=%s instance_compute_patched=%s "
            "graph_owner_attached=%s).",
            type(drafter).__name__,
            getattr(drafter, "method", None),
            getattr(drafter, "parallel_drafting", None),
            getattr(drafter, "_dcut_instance_compute_patched", False),
            getattr(drafter, "_dcut_graph_owner_attached", False),
        )
        self._dcut_logged_drafter_probs = True
