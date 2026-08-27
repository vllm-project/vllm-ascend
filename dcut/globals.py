# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401, SIM105
"""Monkey-patch installer for D-Cut adaptive verifier step-length on **vLLM-Ascend / NPU**.

Ported from the CUDA plugin in ``Bensong0506/vllm`` branch
``feat/dcut-adaptive-verify`` (itself a port of the closed, unmerged vLLM
PR #44885) to run on Huawei Ascend NPU via vllm-ascend (vLLM v0.23.0 base).
The adaptive controller remains a vLLM *general plugin*. This fork also adds
a self-contained 0.23 GDN core plus two D-Cut AscendC state operators.

Algorithm is unchanged (see AngelSlim D-Cut,
https://angelslim.readthedocs.io/zh-cn/latest/dcut.html): the drafter still
proposes ``num_speculative_tokens`` every step, but the verifier only checks a
batch-adaptive subset chosen by a hardware-profiled ITL cost table +
draft-confidence prefix-product scores + batch-wide global top-K.

Only active for parallel speculative methods: ``method=dflash``, or
``method=draft_model`` with ``parallel_drafting=true`` (PARD).

------------------------------------------------------------------------------
GPU -> NPU deltas include two D-Cut state-aware custom operators for GDN
spec-decode. The Python control loop still patches only the NPU runner and
the GDN ``_forward_core`` invoked behind the native custom-op API:

  1. Patch targets: ``NPUModelRunner`` (vllm_ascend.worker.model_runner_v1) /
     ``NPUWorker`` (vllm_ascend.worker.worker) / the Ascend spec-decode
     proposer — NOT the vLLM GPU classes.  This is mandatory: NPUModelRunner
     *overrides* ``execute_model``, ``sample_tokens``,
     ``_copy_draft_token_ids_to_cpu``, ``_update_states`` and ``__init__``, so
     patching ``GPUModelRunner`` would be shadowed by the NPU overrides and the
     plugin would silently no-op.  Likewise ``NPUWorker`` subclasses
     ``WorkerBase`` directly, not ``gpu_worker.Worker``.

  2. Device API: ``torch.cuda.Event`` / ``torch.cuda.synchronize`` ->
     ``torch.npu.Event`` / ``torch.npu.synchronize``.

  3. ``_adaptive_profile_run`` is rebuilt on the NPU forward path
     (``set_ascend_forward_context`` + ``self._model_forward`` + the NPU
     signature of ``_build_attention_metadata``).  The CUDA version relied on
     ``_get_slot_mappings`` / ``_init_model_kwargs`` / a ``slot_mappings=``
     kwarg that **do not exist** on NPUModelRunner, so that path is replaced
     with the NPU ``_dummy_run``-style plumbing.  Profiling is forced eager
     (no ACLGraph capture) — the cost table only needs *relative* ITL.

Enable: ``pip install -e .`` (this dir) + set ``VLLM_DCUT_CONFIG=/path/to.json``
+ ``VLLM_PLUGINS=dcut_adaptive_verify``.  See RUN.md.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from types import MethodType

import numpy as np
import torch

try:  # torch_npu registers the ``torch.npu`` namespace; already imported in a
    # real vllm-ascend worker process, but keep the plugin importable stand-alone.
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from .verify_adaptive_config import VerifyAdaptiveConfig
from .verify_adaptive_controller import VerifyAdaptiveController

logger = init_logger(__name__)

_INSTALLED = False  # WorkerBase deferred-trigger armed (per process)
_PATCHED = False  # real monkey patches applied (per process)
ENV_CONFIG = "VLLM_DCUT_CONFIG"
ENV_DISABLE = "VLLM_DCUT_DISABLE"
ENV_TRIM_STATS_OUT = "VLLM_DCUT_TRIM_STATS_OUT"
ENV_PROFILE_FORCE_EAGER = "VLLM_DCUT_PROFILE_FORCE_EAGER"
ENV_FULL_DECODE_ONLY = "VLLM_DCUT_FULL_DECODE_ONLY"
# Diagnostic-only, non-sensitive boolean flag (default: false). When enabled,
# only the speculative drafter bypasses ACLGraph capture/replay; the target
# verifier keeps its configured graph mode.
ENV_FORCE_DRAFTER_EAGER = "VLLM_DCUT_FORCE_DRAFTER_EAGER"
ENV_GDN_SHARED_STATIC = "VLLM_DCUT_GDN_SHARED_STATIC"

# Adaptive-probability pipeline controls. These are centralized here because
# the D-Cut directory is installed as an independent vLLM general plugin.
# They are non-sensitive runtime tuning flags:
#
# * PROCESS_PROBS_STAGE: "pre_truncate" (default, overlap the previous D2H
#   copy with the rest of the step) or "post_sample" (synchronous baseline).
# * SKIP_UNREADY_PROBS: when true, never wait for the D2H event; reuse the
#   previous cached decision until the copy becomes ready. Default: false.
# * REUSE_ARGMAX: reuse the token IDs already selected by DFlash when deriving
#   selected-token probabilities. Default: true.
ENV_PROCESS_PROBS_STAGE = "VLLM_DCUT_PROCESS_PROBS_STAGE"
ENV_SKIP_UNREADY_PROBS = "VLLM_DCUT_SKIP_UNREADY_PROBS"
ENV_REUSE_ARGMAX = "VLLM_DCUT_REUSE_ARGMAX"

# Compatibility flag used only by the retired patch_gdn.py path. The active
# v0.23 path is controlled by the registered
# VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE variable in vllm_ascend/envs.py.
ENABLE_GDN_MAIN_PIECEWISE_GRAPH = False

# Fixed-address GDN metadata buffers. The active v0.23 graph keys isolate
# target/draft model instances and padded token buckets. Batch metadata is
# shared across GDN layers, while each layer keeps separate state indices.
_dcut_gdn_static = {}
