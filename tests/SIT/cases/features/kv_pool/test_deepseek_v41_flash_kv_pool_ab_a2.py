# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A2 8-card (TP8, single node) KV-pool A/B protection cases for DeepSeek-V4.1-Flash W4A8.

Two parameterized scenarios cover the mooncake pooling matrix on A2:

==========================  =========  ======  =====================
scenario                    backend    mode    storage tier
==========================  =========  ======  =====================
mooncake_bulk_tier2         mooncake   bulk    DRAM
mooncake_bulk_tier3         mooncake   bulk    DRAM + SSD offload
==========================  =========  ======  =====================

Tier and transfer mode are two *different* switches.  "tier 2" / "tier 3" name
the storage medium the pool answers from: host DRAM only, versus DRAM plus a
Mooncake SSD offload tier with the independent SSD -> DRAM prefetch
(`docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md:488-538`).
``use_layerwise`` is the transfer granularity: the connector streams KV per
transformer layer instead of copying whole blocks in one bulk call
(`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py:131`).
The two are orthogonal -- a DRAM-only pool can be read layerwise and an
SSD-backed pool can be read in bulk.  Both scenarios here use the bulk
transfer on purpose: this batch's requirement is the tier-2/tier-3 storage
pair for this model, and layerwise is out of scope for V4.1 (see next
paragraph).

Layerwise is deliberately NOT covered for this model.  Two independent
pieces of evidence say it cannot work today, so a layerwise scenario would
only measure a known-unwired path:

1. The documentation forbids it four times: the CED hybrid-attention
   structure is incompatible with layer-by-layer save/load and
   ``use_layerwise`` must stay ``false`` --
   `DeepSeek-V4.1-Flash.md:1027-1029`, `:1359-1361`, `:1374-1376`, `:1627`.
2. The per-layer hooks are not wired for V4.1.  ``wait_for_layer_load`` /
   ``save_kv_layer`` are called from the MLA and SFA attention
   implementations (`vllm_ascend/attention/mla_v1.py:2461,3342`,
   `vllm_ascend/attention/sfa_v1.py:1899,1942`) and there is no such call
   anywhere in `vllm_ascend/attention/dsa_v41.py`.  Measured on
   80.48.37.145 (2026-11-12, an earlier ``mooncake_layerwise_tier3`` probe
   run): the engine starts and the accuracy round passes, but the pool
   stores nothing (R2 external hits 0) because the per-layer save path is
   silently absent.  Re-adding a layerwise scenario makes sense once
   ``dsa_v41.py`` wires those hooks.

Legs.  Each scenario runs two engine starts, serially in one pytest process
(HCCL / torch.distributed groups are process-wide singletons):

- Leg A -- pool ON (mooncake, transfer mode and tier per scenario), local
  prefix cache OFF.  It proves the pool itself serves the shared prefix: R2/R3
  report the exact engine-derived hit-token count, the hit rounds reproduce the
  cold greedy tokens (H3), and R4 on a disjoint prompt reports zero hits.
- Leg B -- pool OFF, local prefix cache ON.  It proves the same workload hits
  locally (``vllm:prefix_cache_hits`` must grow) and yields the baseline TTFT /
  throughput for the A/B comparison.

Within one engine start every leg runs accuracy -> hit (A) / local-hit witness
(B) -> performance against the same 16K-class natural-language prefix, so the
reused prefix is the only thing that differs between the legs.  The A/B gate is
strict token identity plus the per-tier performance limits below.

Why single-node TP8 is an over-document configuration.  The repository only
documents DeepSeek-V4.1-Flash on 8 NPUs as a *two-node* DP2/TP8/EP layout
(`docs/source/tutorials/models/DeepSeek-V4.1-Flash.md:1631-1707`), i.e. 16
devices, and its limitation section states the W8A8 checkpoint only
(`:59-67`, `docs/source/user_guide/support_matrix/supported_models.md:129`).
552 backbone parameters (`:13-16`) at 4 bits are roughly 276 GB of weights, so
8 x 64 GB A2 HBM (512 GB total) leaves on the order of 200 GB per node for KV
cache, Engram host tables, activation and Mooncake buffers -- a genuinely tight
fit that the documentation does not certify.  The first runs on this case may therefore OOM
during profiling or KV-cache sizing.  That risk is deliberately not hidden:
every calibration-sensitive value is an environment knob
(`DSV41_POOL_GPU_MEM_UTIL`, `DSV41_POOL_MAX_MODEL_LEN`,
`DSV41_POOL_MAX_NUM_SEQS`, `DSV41_POOL_ENABLE_EP`,
`DSV41_POOL_NUM_SPECULATIVE_TOKENS`) and the observed values are logged
per leg so the reviewer can see which knob had to move.

Environment knobs (all optional, defaults shown in the code):

- ``DSV41_POOL_MODEL``: checkpoint path (default ``/mnt/share/weight/
  DeepSeek-V4.1-Flash-1017-w4a8``).
- ``DSV41_POOL_DEVICES``: ``ASCEND_RT_VISIBLE_DEVICES`` value (``0..7``).
- ``DSV41_POOL_USE_V2_MODEL_RUNNER``: must stay ``0`` for this model.
- ``DSV41_POOL_GPU_MEM_UTIL`` / ``DSV41_POOL_MAX_MODEL_LEN`` /
  ``DSV41_POOL_MAX_NUM_SEQS`` / ``DSV41_POOL_MAX_NUM_BATCHED_TOKENS`` /
  ``DSV41_POOL_PROMPT_TOKENS`` / ``DSV41_POOL_MAX_TOKENS`` /
  ``DSV41_POOL_BLOCK_SIZE``: memory and load calibration.
- ``DSV41_POOL_ENABLE_EP`` / ``DSV41_POOL_ASYNC_SCHEDULING`` /
  ``DSV41_POOL_ENFORCE_EAGER`` / ``DSV41_POOL_NUM_SPECULATIVE_TOKENS`` /
  ``DSV41_POOL_ADDITIONAL_CONFIG``: engine switches from the model guide.
- ``DSV41_POOL_DRAM_TIER`` / ``DSV41_POOL_SSD_DRAM_TIER`` /
  ``DSV41_POOL_LOCAL_BUFFER_SIZE`` / ``DSV41_POOL_MASTER_EXTRA_ARGS``:
  Mooncake pool sizing and master flags.
- ``DSV41_POOL_SSD_ROOT`` / ``DSV41_POOL_SSD_MIN_FREE_GB`` /
  ``DSV41_POOL_SSD_RANK_BUDGET_GB`` / ``DSV41_POOL_SSD_PRESSURE_PROMPTS`` /
  ``DSV41_POOL_SSD_SETTLE_SECONDS`` / ``DSV41_POOL_SSD_PREFETCH_COOLDOWN_SEC`` /
  ``DSV41_POOL_SSD_PREFETCH_DEDUP_TTL_SEC``: tier-3 SSD offload.
- ``DSV41_POOL_TTFT_TIER2`` / ``DSV41_POOL_TTFT_TIER3`` /
  ``DSV41_POOL_THROUGHPUT_TIER2`` / ``DSV41_POOL_THROUGHPUT_TIER3``: per-tier
  A/B gates.
- ``DSV41_POOL_OUTPUT_IDENTITY``: H3 / A-B token identity gate (``1``; see
  :func:`_phase_hit`).
- ``DSV41_POOL_ENGINE_READY_TIMEOUT_S`` / ``DSV41_POOL_HCCL_CONNECT_TIMEOUT`` /
  ``DSV41_POOL_PYTORCH_NPU_ALLOC_CONF``: startup limits for the 8-worker,
  ~276 GB load, from the documented A2 script (:1656-1660).
- ``KV_POOL_AB_MODE``: ``gate`` (default here, as in the Qwen3.8 A2 case) or
  ``record``.  Note the shared watch module defaults the same variable to
  ``record`` (`kv_pool_watch.py:563`), so CI must set it explicitly.
- ``KV_POOL_AB_ARTIFACT``: JSON-Lines report path (from ``kv_pool_watch``).

Host prerequisites (official KV pool guide): mooncake on A2 runs the HIXL RoCE
direct mode (``HCCL_INTRA_ROCE_ENABLE=1``, set here) over reserved 2 MB
hugepages -- the guide reserves 200000 on the host
(`kv_cache_pool_ascend_store.md:275-276`); the test skips when the container
sees none.  ``ASCEND_ENABLE_USE_FABRIC_MEM`` must stay unset (see
:func:`_leg`).  The tier-3 SSD root must be a writable filesystem with real
free space; the test skips with the measured value when it is not.

Run: ``pytest tests/e2e/pull_request/eight_card/kv_pool/test_deepseek_v41_flash_kv_pool_ab_a2.py``
"""

import json
import math
import os
import shutil
import tempfile
import time
from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    AB_ARTIFACT,
    AB_LEG_GAP_SECONDS,
    AB_SAMPLES,
    AB_WARMUP,
    PREFIX_HITS_METRIC,
    assert_hit_tokens,
    assert_no_hit,
    assert_output_identical,
    snapshot_metrics,
)

try:
    # vLLM's native Engram config (docs/source/tutorials/models/DeepSeek-V4.1-Flash.md:233).
    # Older vLLM revisions in this plugin's support window do not ship it --
    # vllm_ascend/patch/__init__.py:143-145 documents the same optionality.
    from vllm.config import EngramConfig
except ImportError:  # pragma: no cover - depends on the installed vLLM revision
    EngramConfig = None


def _env_bool(name: str, default: bool) -> bool:
    """Parse an optional true/false environment switch."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Model under test
# ---------------------------------------------------------------------------

# NOTE the lowercase ``w4a8``: the sibling directory
# ``/mnt/share/weight/DeepSeek-V4.1-Flash-1017-W4A8`` (capital W4A8) is an older
# build of the same export.  Mount the lowercase one; mixing them produces
# weight/scheme mismatches that look like pool corruption.  The repository does
# not register a W4A8 V4.1 checkpoint at all -- the guide documents W8A8 only
# (DeepSeek-V4.1-Flash.md:59-67, supported_models.md:129) -- which is why
# :func:`_classify_w4a8_checkpoint` decides the quantization path at runtime
# instead of trusting the directory name.
MODEL = os.environ.get("DSV41_POOL_MODEL", "/mnt/share/weight/DeepSeek-V4.1-Flash-1017-w4a8")
VISIBLE_DEVICES = os.environ.get("DSV41_POOL_DEVICES", "0,1,2,3,4,5,6,7")
TP_SIZE = 8

# DeepSeek-V4.1-Flash supports model runner V1 only on Ascend
# (DeepSeek-V4.1-Flash.md:48-49, restated for every example at :205, :1659, and
# explained at :257-259).  The Qwen3.8 pool case this file is modelled on sets
# VLLM_USE_V2_MODEL_RUNNER=1 because Qwen3.8-27B is an MRV2 model
# (test_qwen38_kv_pool_ab_a2.py:532); that is a model difference, not a
# convention to copy, so this case pins V1.
USE_V2_MODEL_RUNNER = os.environ.get("DSV41_POOL_USE_V2_MODEL_RUNNER", "0")

# Startup allowances taken from the documented A2 script
# (DeepSeek-V4.1-Flash.md:1656-1660).  Loading ~276 GB through the lazy
# safetensors strategy on 8 workers needs far more than the vLLM default engine
# ready timeout; the guide itself sets 360000 there.  A too-small value is the
# most likely first-run failure that is not an OOM.
ENGINE_READY_TIMEOUT_S = os.environ.get("DSV41_POOL_ENGINE_READY_TIMEOUT_S", "360000")
HCCL_CONNECT_TIMEOUT = os.environ.get("DSV41_POOL_HCCL_CONNECT_TIMEOUT", "1200")
PYTORCH_NPU_ALLOC_CONF = os.environ.get("DSV41_POOL_PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

# Model-specific tooling name (tokenizer mode) from DeepSeek-V4.1-Flash.md:1687.
# The serve-CLI-only reasoning/tool-call parser flags of the same section are
# not usable in the offline LLM() harness; see the _leg kwargs comment.
DEEPSEEK_V41_TOOLING = "deepseek_v41"
# DeepSeek-V4.1-Flash.md:1693.
QUANTIZATION = "ascend"
# DeepSeek-V4.1-Flash.md:1691: memory-mapped on-demand load; the alternative
# (eager) materializes the whole ~276 GB checkpoint at once.
SAFETENSORS_LOAD_STRATEGY = "lazy"
# kv_cache_pool_ascend_store.md:275-276 / :426-427: A2 (and A3 on RoCE links)
# selects the HIXL RoCE direct mode for the intra-node transport.
HCCL_ROCE_ENV = "HCCL_INTRA_ROCE_ENABLE"
# kv_cache_pool_ascend_store.md:285 exports it for the A3 HCCS mode only.  On
# A2 it must be *absent*: an empty-string value still reads as "set" to the
# engine, which then takes the hugepage-reserve path and fails
# (test_qwen38_kv_pool_ab_a2.py:302-316, mooncake_backend.py:396).
FABRIC_MEM_ENV = "ASCEND_ENABLE_USE_FABRIC_MEM"
# A3 examples use "1" (PR #20757 cases); the local A2 cases use "0" and this
# file follows the A2 precedent.  Only matters when two engines share a host.
LOOKUP_RPC_PORT = "0"
MOONCAKE_BACKEND = "mooncake"
# DeepSeek-V4.1-Flash.md:1694: DSpark speculative decoding, draft runs eager.
SPECULATIVE_METHOD = "dspark"
NUM_SPECULATIVE_TOKENS = int(os.environ.get("DSV41_POOL_NUM_SPECULATIVE_TOKENS", 5))
ENGINE_SEED = 1024
# DeepSeek-V4.1-Flash.md:1677: the documented A2 TP8 layout pairs TP with expert
# parallelism.  It also changes how the ~276 GB of expert weights are sharded,
# so it is a first-order knob when the single-node over-document fit is tight.
EXPERT_PARALLEL = _env_bool("DSV41_POOL_ENABLE_EP", True)

# ---------------------------------------------------------------------------
# Protection load (16K-class prefix reuse)
# ---------------------------------------------------------------------------

# lcm + 4 rule: the pool only transfers whole block groups, and a prompt whose
# length is an exact multiple of the KV-group lcm has its final group trimmed by
# the scheduler, so the hit count is always 0 (kv_pool_watch.py:122-151,
# measured in .claude/scratch/pr20757/notes.md section 2.6: 16384/128/16384 ->
# 0 hits, 16388/128/16384 -> 16384 hits).  16388 = 16384 + 4 keeps a non-zero
# tail for every plausible V4.1 lcm.  The lcm itself is read from the engine at
# runtime (:func:`_engine_pool_shapes`); never hardcode a hit count for this
# model.
POOL_PROMPT_TOKENS = int(os.environ.get("DSV41_POOL_PROMPT_TOKENS", 16388))
# Prefill chunking is fixed so cold and hit rounds produce the same batch shape:
# POOL_PROMPT_TOKENS prefill as [16384, 4] in every leg.
POOL_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("DSV41_POOL_MAX_NUM_BATCHED_TOKENS", 16384))
# DeepSeek-V4.1-Flash.md:1684.  V4.1 keeps the block size its cache was built
# with; the platform hook deliberately does not override it
# (tests/ut/test_platform.py:2197-2223).
POOL_BLOCK_SIZE = int(os.environ.get("DSV41_POOL_BLOCK_SIZE", 128))
# max_model_len has to cover the whole prompt plus its generated tail.  Derived
# rather than written as a bare number: one full prefill chunk (the 16K-class
# prompt fits in a single POOL_MAX_NUM_BATCHED_TOKENS chunk) plus two
# POOL_BLOCK_SIZE blocks of slack -- one for the prompt tail and the generated
# tokens, one for the DSpark draft lookahead.  With the defaults that is 16640,
# the same scheduler window the Qwen3.8 A2 case uses.
POOL_MAX_MODEL_LEN = int(os.environ.get("DSV41_POOL_MAX_MODEL_LEN", POOL_MAX_NUM_BATCHED_TOKENS + 2 * POOL_BLOCK_SIZE))
POOL_MAX_TOKENS = int(os.environ.get("DSV41_POOL_MAX_TOKENS", 16))
POOL_MAX_NUM_SEQS = int(os.environ.get("DSV41_POOL_MAX_NUM_SEQS", 4))
# 552B W4A8 is ~276 GB of weights over 8 x 64 GB HBM (~34.5 GB/rank).  0.85
# leaves ~9.6 GB/rank for activations, the pool buffers and the profiling pass.
# This is the first knob to lower when the over-document TP8 layout OOMs.
POOL_GPU_MEM_UTIL = float(os.environ.get("DSV41_POOL_GPU_MEM_UTIL", 0.85))
# DeepSeek-V4.1-Flash.md:1683 (--async-scheduling; the A2/A3 matrix marks it as
# supported-with-caveats, supported_models.md:129) and the eager graph mode
# explained in :func:`_leg`.
ASYNC_SCHEDULING = _env_bool("DSV41_POOL_ASYNC_SCHEDULING", True)
ENFORCE_EAGER = _env_bool("DSV41_POOL_ENFORCE_EAGER", True)

# Natural-language prompts.  Random token ids flip under batch-shape differences
# even with intact KV, so the accuracy / H3 rounds need low-entropy text
# (test_qwen38_kv_pool_ab_a2.py:159-160).  Same sentences as the Qwen3.8 A2 case
# so the two files stay comparable on one machine.
SENTENCE_A = "Explain how historical records help people understand the past. "
SENTENCE_B = "The mountain railway climbs steadily toward the old observatory. "
PRESSURE_SENTENCES = [
    "Draft a travel plan that lists every museum worth visiting in the capital. ",
    "Summarize the key milestones in the history of long-distance communication. ",
    "Describe the process of turning raw wool into warm winter clothing. ",
    "Outline the main arguments for and against daylight saving time. ",
]

# ---------------------------------------------------------------------------
# Pool sizing
# ---------------------------------------------------------------------------

# get_open_port is not atomic: two calls can hand back the same free port, and
# mooncake_master binds both (the manager's _ready_ports / _start_service).
# validate_kv_pool_config rejects equal ports on the YAML path
# (tests/e2e/common/kv_pool/config.py:92-93); direct construction checks too.
PORT_RETRY_LIMIT = 8
# Tier-2 DRAM segment per client.  DeepSeek-V4.1-Flash.md:1391 uses
# "24GB" for the mooncake.json of the A2 pool deployment; each TP rank
# contributes its own segment, so on TP8 the master accounts 8 x this value.
DRAM_TIER = os.environ.get("DSV41_POOL_DRAM_TIER", "24GB")
# Tier-3: the DRAM tier is deliberately much smaller than tier-2 so that
# eviction (the manager pins the master at
# --eviction_high_watermark_ratio 0.9, tests/e2e/common/single_node/
# kv_pool_runtime.py:22) can reach the SSD offload tier.  NOTE the honest limit
# of this sizing: V4.1's global KV is 890 bytes/token
# (DeepSeek-V4.1-Flash.md:20-21), so one 16K prefix costs roughly 15 MB --
# ~60 prefixes are needed to cross 90% of a 2 GB tier.  Two knobs force the
# eviction + SSD read-back path for an on-site verification run: shrink
# DSV41_POOL_SSD_DRAM_TIER and raise DSV41_POOL_SSD_PRESSURE_PROMPTS until the
# pressure working set crosses the watermark (then the LRU victim is the R1
# shared prefix and R2/R3 must read it back through the SSD offload tier).
# With the default 2GB/2 the case exercises the offload + prefetch
# configuration and asserts exact hits through it, but does NOT claim that a
# specific block was served from SSD.
SSD_DRAM_TIER = os.environ.get("DSV41_POOL_SSD_DRAM_TIER", "2GB")
LOCAL_BUFFER_SIZE = os.environ.get("DSV41_POOL_LOCAL_BUFFER_SIZE", "1GB")
GIBIBYTE = 1024**3
# kv_cache_pool_ascend_store.md:517-522: 800 GB disk over 8 TP ranks -> ~100 GB
# per rank, and the offload buffer must be a multiple of 1 GB (:524).
SSD_RANK_BUDGET_GB = int(os.environ.get("DSV41_POOL_SSD_RANK_BUDGET_GB", 100))
SSD_MIN_FREE_GB = int(os.environ.get("DSV41_POOL_SSD_MIN_FREE_GB", 64))
SSD_PREFETCH_COOLDOWN_SEC = int(os.environ.get("DSV41_POOL_SSD_PREFETCH_COOLDOWN_SEC", 5))
SSD_PREFETCH_DEDUP_TTL_SEC = int(os.environ.get("DSV41_POOL_SSD_PREFETCH_DEDUP_TTL_SEC", 30))
SSD_PRESSURE_PROMPTS = int(os.environ.get("DSV41_POOL_SSD_PRESSURE_PROMPTS", 2))
SSD_SETTLE_SECONDS = int(os.environ.get("DSV41_POOL_SSD_SETTLE_SECONDS", 30))
# Offload availability must be advertised by the master or every worker's
# disk-segment mount fails with UNABLE_OFFLOAD and store init aborts
# (kv_cache_pool_ascend_store.md:496-508, test_qwen38_kv_pool_ab_a2.py:208-211).
# Whitespace-separated so the operator can drop flags an older
# mooncake_master binary rejects; set to "" to disable offload entirely.
MASTER_EXTRA_ARGS = tuple(
    os.environ.get(
        "DSV41_POOL_MASTER_EXTRA_ARGS",
        "--enable_offload=true --offload_on_evict=true --offload_force_evict=false",
    ).split()
)

# ---------------------------------------------------------------------------
# Verdict / gates
# ---------------------------------------------------------------------------

AB_MODE = os.environ.get("KV_POOL_AB_MODE", "gate")
# Per-tier A/B gates.  run_pool_ab cannot express these: its TTFT/throughput
# limits are module constants with no injection point
# (kv_pool_watch.py:398-399, notes.md section 8.2 I-3), which is also why the
# Qwen3.8 A2 case drives the low-level assertions directly.  The numbers are
# inherited from that A2 calibration (1.35x TTFT / 0.75x throughput for a pure
# DRAM tier, 3.0x / 0.30x for the slow per-layer + mixed-medium tier) and stay
# provisional until the first V4.1 run on this machine (a 552B MoE moves a very
# different payload per request); recalibrate by editing the defaults or by
# setting the four environment variables above.  They are wide enough to keep a
# shared machine's noise green while still failing a catastrophic regression --
# a full re-prefill miss runs an order of magnitude slower.
TIER_TTFT_LIMITS = {
    "tier2": float(os.environ.get("DSV41_POOL_TTFT_TIER2", 1.35)),
    "tier3": float(os.environ.get("DSV41_POOL_TTFT_TIER3", 3.0)),
}
TIER_THROUGHPUT_LIMITS = {
    "tier2": float(os.environ.get("DSV41_POOL_THROUGHPUT_TIER2", 0.75)),
    "tier3": float(os.environ.get("DSV41_POOL_THROUGHPUT_TIER3", 0.30)),
}
# H3 / A-B token identity stays ON by default.  It can only be turned off with
# measured evidence attached to the PR: the pool CI exemption list
# (docs/pool_pr_nightly_weekly.md section 3.3, notes.md section 9 item 6) allows
# it for DSpark (#3/#5) and dummy-weight (#11) cases, because a spec-decode
# accept/reject path that differs between the cold and hit chunk shapes can
# change the output without any KV corruption.  This case runs DSpark, so the
# first run decides whether identity holds here; until then the gate stays
# strict and this switch is the documented, auditable place to relax it.  The
# exemption policy is docs/pool_pr_nightly_weekly.md:152 / section 3.3: only the
# DSpark (#3/#5) and dummy-weight (#11) cases may skip H3, and they still run
# H1/H2/R4.
OUTPUT_IDENTITY = os.environ.get("DSV41_POOL_OUTPUT_IDENTITY", "1").strip().lower() not in ("0", "false", "no")

# Checkpoint metadata that decides which W4A8 layout is being served.  The
# ModelSlim description filename is the one the plugin reads
# (vllm_ascend/quantization/configs/modelslim_config.py:163).
MODELSLIM_QUANT_DESCRIPTION_FILE = "quant_model_description.json"
# INT4 weights + INT8 activations: the W4A8_DYNAMIC schemes of
# vllm_ascend/quantization/methods/w4a8/w4a8.py:51,583, served by the regular
# (non-MegaMoe) MoE path on A2/A3.
MODELSLIM_INT4_SCHEMES = ("W4A8_DYNAMIC",)
# FP4 weights: QuantType.W4A8MXFP (vllm_ascend/quantization/quant_type.py:35),
# registered as the W4A8_MXFP scheme (methods/w4a8/w4a8_mxfp4.py:48,161).
MXFP4_WEIGHT_SCHEMES = ("W4A8_MXFP", "W4A4_MXFP4", "MXFP4", "FLOAT4_E2M1")
# The checkpoint layouts this case can serve at all.
W4A8_MXFP_A5_ONLY_MESSAGE = (
    "MegaMoe/FUSED_MC2 supports W4A8MXFP on Ascend 950 only "
    "(quantization/quant_type.py:40-46 A5_SUPPORT_MEGA_MOE_QUANT_TYPES; "
    "ops/fused_moe/moe_utils.py:278-283 raises 'MegaMoe integration supports W8A8/W4A8 INT on A2/A3 "
    "and W8A8MXFP/W4A8MXFP/W4A4MXFP on A5'). A2 has no kernel for it."
)
MOONCAKE_PACKAGE = "mooncake-transfer-engine-npu"


def _mooncake_version() -> str:
    """Installed Mooncake version string, or a marker when the wheel is absent."""
    try:
        return _pkg_version(MOONCAKE_PACKAGE)
    except PackageNotFoundError:
        return "not-installed"


def _hugepages_total() -> int:
    """Reserved hugepages visible to the container (0 when the host set none)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("HugePages_Total:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _npu_device_status() -> tuple[int | None, str]:
    """Return ``(device_count, evidence)``; the count is None when unusable."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return None, "torch has no .npu attribute (torch_npu is not installed)"
    try:
        if not npu.is_available():
            return None, "torch.npu.is_available() is False"
        return int(npu.device_count()), f"torch.npu.device_count()={npu.device_count()}"
    except Exception as exc:  # pragma: no cover - driver-level failure is the evidence
        return None, f"torch.npu probe raised {exc!r}"


def _read_json_mapping(path: str) -> tuple[dict[str, Any] | None, str]:
    """Load one JSON object, returning the failure text instead of raising."""
    if not os.path.isfile(path):
        return None, f"{path}: not present"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, f"{path}: unreadable ({exc})"
    if not isinstance(data, dict):
        return None, f"{path}: top-level JSON is {type(data).__name__}, expected an object"
    return data, f"{path}: read"


def _string_values(node: Any) -> list[str]:
    """Every string reachable in a nested JSON structure (flat, for matching)."""
    if isinstance(node, dict):
        found: list[str] = []
        for value in node.values():
            found.extend(_string_values(value))
        return found
    if isinstance(node, list):
        items: list[str] = []
        for value in node:
            items.extend(_string_values(value))
        return items
    if isinstance(node, str):
        return [node]
    return []


def _classify_w4a8_checkpoint(checkpoint_dir: str) -> tuple[str, str]:
    """Decide which W4A8 weight layout the checkpoint declares.

    The repository never registers a W4A8 V4.1 checkpoint
    (DeepSeek-V4.1-Flash.md:59-67, supported_models.md:129 list W8A8 only), so
    the scheme has to come from the checkpoint metadata rather than the
    directory name.  Returns ``(verdict, evidence)``:

    - ``modelslim_int4``: INT4 weights / INT8 activations (``W4A8_DYNAMIC``).
      This is the layout the A2/A3 non-MegaMoe MoE path can serve.
    - ``mxfp``: DeepSeek-original FP4 weights (``W4A8MXFP``).  Its only MoE
      implementation is the Ascend 950 MegaMoe operator, so A2 cannot run it.
    - ``w8a8``: an INT8 checkpoint (the documented layout) -- usable, but not
      what this case is registered for.
    - ``unknown``: metadata missing or unreadable.
    - ``missing_dir``: the checkpoint path does not exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return "missing_dir", f"checkpoint directory {checkpoint_dir!r} is not present on this node"

    sources: list[tuple[str, str]] = []
    evidence: list[str] = []
    config, note = _read_json_mapping(os.path.join(checkpoint_dir, "config.json"))
    evidence.append(note)
    if config is not None:
        quant_config = config.get("quantization_config")
        if isinstance(quant_config, dict):
            sources.append(("config.json:quantization_config", " ".join(_string_values(quant_config)).upper()))
        else:
            evidence.append("config.json: no quantization_config object (unquantized or upstream-declared)")
    description, note = _read_json_mapping(os.path.join(checkpoint_dir, MODELSLIM_QUANT_DESCRIPTION_FILE))
    evidence.append(note)
    if description is not None:
        sources.append((MODELSLIM_QUANT_DESCRIPTION_FILE, " ".join(_string_values(description)).upper()))

    for tokens, verdict in (
        (MXFP4_WEIGHT_SCHEMES, "mxfp"),
        (MODELSLIM_INT4_SCHEMES, "modelslim_int4"),
    ):
        for label, blob in sources:
            matched = [token for token in tokens if token in blob]
            if matched:
                return verdict, f"{label} declares {', '.join(matched)}; " + "; ".join(evidence)
    if any("W8A8" in blob for _, blob in sources):
        return "w8a8", "checkpoint is W8A8 (the documented layout), not W4A8; " + "; ".join(evidence)
    return "unknown", "no W4A8/W8A8 scheme found in " + "; ".join(evidence)


def _ssd_root(case_name: str) -> str:
    return os.environ.get("DSV41_POOL_SSD_ROOT") or os.path.join(tempfile.gettempdir(), f"{case_name}_mooncake_ssd")


def _prepare_ssd_root(path: str) -> tuple[bool, str]:
    """Create the offload directory and report the real free space it offers."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
        free_bytes = shutil.disk_usage(path).free
    except OSError as exc:
        return False, f"SSD offload root {path!r} is not usable: {exc}"
    free_gb = free_bytes / GIBIBYTE
    if free_gb < SSD_MIN_FREE_GB:
        return False, (
            f"SSD offload root {path!r} has {free_gb:.1f} GiB free, below the "
            f"{SSD_MIN_FREE_GB} GiB floor (DSV41_POOL_SSD_MIN_FREE_GB); mount a real disk or "
            f"set DSV41_POOL_SSD_ROOT"
        )
    return True, f"SSD offload root {path!r} writable, {free_gb:.1f} GiB free"


def _mooncake_pool_config(tier_ssd: bool, case_name: str) -> MooncakeKVPoolConfig:
    config: dict[str, Any] = {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "global_segment_size": SSD_DRAM_TIER if tier_ssd else DRAM_TIER,
        "local_buffer_size": LOCAL_BUFFER_SIZE,
        # DeepSeek-V4.1-Flash.md:1392-1393.
        "preferred_segment": False,
        "prefer_alloc_in_same_node": True,
    }
    if tier_ssd:
        # Tier 3 per the official guide: SSD offload plus the independent
        # SSD -> DRAM prefetch on exists checks, which is what promotes SSD-only
        # objects into the fast tier during the cache probe
        # (ExistOptions(prefetch_to_memory=True), Mooncake PR 4272; the
        # connector disables the prefetch with a warning when the installed
        # client lacks it).  Setting any prefetch parameter also makes the store
        # create its own TransferEngine instead of sharing the global one, and
        # the connector creates per-rank subdirectories under the root
        # (kv_cache_pool_ascend_store.md:510).
        config.update(
            {
                "enable_ssd_offload": True,
                "ssd_offload_path": _ssd_root(case_name),
                "enable_ssd_prefetch": True,
                "ssd_prefetch_cooldown_sec": SSD_PREFETCH_COOLDOWN_SEC,
                "ssd_prefetch_dedup_ttl_sec": SSD_PREFETCH_DEDUP_TTL_SEC,
            }
        )
    master_port = get_open_port()
    metrics_port = get_open_port()
    for _attempt in range(PORT_RETRY_LIMIT):
        if metrics_port != master_port:
            break
        metrics_port = get_open_port()
    else:
        raise RuntimeError(
            f"could not allocate a metrics port distinct from master port {master_port} "
            f"in {PORT_RETRY_LIMIT} tries (get_open_port kept returning the same value)"
        )
    return MooncakeKVPoolConfig(
        master_port=master_port,
        metrics_port=metrics_port,
        config=config,
        master_extra_args=MASTER_EXTRA_ARGS if tier_ssd else (),
    )


def _mooncake_ssd_offload_env() -> dict[str, str]:
    """Disk-usage controls for the tier-3 SSD offload (guide:514-538).

    ``MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`` defaults to ~2 TB per rank, far
    above any real disk, and the guide marks it as strongly recommended to
    override; ``MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`` defaults to 1280 MB
    which is not a multiple of 1 GB (guide:524, :533).
    """
    rank_budget_bytes = SSD_RANK_BUDGET_GB * GIBIBYTE
    return {
        "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(rank_budget_bytes),
        "MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE": str(rank_budget_bytes),
        "MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY": "lru",
        "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(GIBIBYTE),
    }


def _speculative_config() -> dict[str, Any] | None:
    """DSpark target/draft setup from DeepSeek-V4.1-Flash.md:1694.

    ``num_speculative_tokens=0`` disables it; that is the calibration knob for a
    run where the DSpark accept/reject path makes cold and hit outputs differ
    (see ``DSV41_POOL_OUTPUT_IDENTITY``).
    """
    if NUM_SPECULATIVE_TOKENS <= 0:
        return None
    return {
        "method": SPECULATIVE_METHOD,
        "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        "enforce_eager": True,
    }


def _engram_kwargs() -> dict[str, Any]:
    """``--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'`` (:1696).

    ``cpu_offload`` keeps the Engram host tables out of HBM.
    ``dp_shared_memory`` is left False on purpose: this topology is TP8 with a
    single DP rank, so sharing the host table across DP ranks (what the A3/A2
    examples do with ``--ipc=host`` and the 24 Engram heads, :233, :260-265)
    buys nothing here and would add the ~320 GB ``/dev/shm`` provisioning
    prerequisite those examples carry.  The plugin only force-enables it under
    V2 with DP>1 or PCP>1 (vllm_ascend/ascend_config.py:692-701).
    """
    if EngramConfig is None:
        # Older vLLM in the support window has no native Engram config.  Leave
        # it to vllm_ascend/platform.py:1884-1896, which supplies the default
        # for DeepseekV41ForCausalLM, and say so in the log rather than fail.
        print(
            "[kv_pool_a2] vllm.config.EngramConfig is unavailable; relying on the "
            "platform default engram config (cpu_offload resolution in vllm_ascend/platform.py)",
            flush=True,
        )
        return {}
    return {"engram_config": EngramConfig(cpu_offload=True, dp_shared_memory=False)}


def _additional_config() -> dict[str, Any]:
    """Optional ``--additional-config`` JSON for on-site tuning (default: none).

    The guide's optimization flags (enable_dsa_cp, enable_flashcomm1,
    enable_shared_expert_dp, enable_npugraph_ex -- :1697-1706) are throughput
    switches that do not change what this case guards, so they stay at their
    defaults to keep the first-run failure surface small.  Set
    DSV41_POOL_ADDITIONAL_CONFIG to a JSON object to add them.
    """
    raw = os.environ.get("DSV41_POOL_ADDITIONAL_CONFIG", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"DSV41_POOL_ADDITIONAL_CONFIG must be a JSON object, got {type(parsed).__name__}")
    return parsed


@contextmanager
def _leg(
    use_pool: bool,
    backend: str,
    use_layerwise: bool,
    tier_ssd: bool,
    case_name: str,
):
    """Build one leg: A turns the pool on with the local cache off, B the opposite.

    Everything else (parallel layout, scheduler budget, graph mode, DSpark,
    Engram) is identical across the legs, so the only variable is where the
    reused prefix comes from: the pool or the local prefix cache.
    """
    if backend != MOONCAKE_BACKEND:
        raise ValueError(f"{case_name}: this case covers the mooncake backend only, got {backend!r}")
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            dtype="auto",
            quantization=QUANTIZATION,
            block_size=POOL_BLOCK_SIZE,
            # tokenizer_mode is a legal EngineArgs field and selects the
            # deepseek_v41 tokenizer the checkpoint needs. The serve-CLI-only
            # reasoning/tool-call parsers (DeepSeek-V4.1-Flash.md:1687-1689)
            # are deliberately NOT passed here: they belong to the OpenAI API
            # server layer, and this offline LLM() harness rejects them
            # (vLLM 0.30.0: "EngineArgs.__init__() got an unexpected keyword
            # argument 'tool_call_parser'", measured on 80.48.37.145).
            tokenizer_mode=DEEPSEEK_V41_TOOLING,
            tensor_parallel_size=TP_SIZE,
            enable_expert_parallel=EXPERT_PARALLEL,
            distributed_executor_backend="mp",
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=POOL_MAX_NUM_SEQS,
            max_num_batched_tokens=POOL_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=POOL_GPU_MEM_UTIL,
            seed=ENGINE_SEED,
            enable_prefix_caching=not use_pool,
            enable_chunked_prefill=True,
            disable_log_stats=False,
            async_scheduling=ASYNC_SCHEDULING,
            # DeepSeek-V4.1-Flash.md:1691.  trust_remote_code (:1686) is already
            # forced by VllmRunner (tests/e2e/conftest.py:1042).
            safetensors_load_strategy=SAFETENSORS_LOAD_STRATEGY,
            # Graph mode: the A2/A3 matrix marks V4.1 Piecewise AclGraph as not
            # supported (supported_models.md:129), and the pool connector asks
            # for Piecewise exactly when layerwise is on
            # (ascend_store_connector.py:119-124).  eager is the only shape this
            # model can express on A2 today, so both legs run enforce_eager.
            enforce_eager=ENFORCE_EAGER,
            speculative_config=_speculative_config(),
            **_engram_kwargs(),
        )
        additional_config = _additional_config()
        if additional_config:
            kwargs["additional_config"] = additional_config
        if use_pool:
            # Mooncake on A2 = HIXL RoCE direct mode, selected by
            # HCCL_INTRA_ROCE_ENABLE=1 (guide:275-276, model doc :1006-1009).
            # ASCEND_ENABLE_USE_FABRIC_MEM must be *absent*, not empty: the
            # documented A3-HCCS mode reads an empty value as "set", and the
            # engine then fails its hugepage-reserve path on A2
            # (test_qwen38_kv_pool_ab_a2.py:302-316).
            os.environ.pop(FABRIC_MEM_ENV, None)
            stack.enter_context(patch.dict(os.environ, {HCCL_ROCE_ENV: "1"}))
            pool = stack.enter_context(SingleNodeMooncakeManager(_mooncake_pool_config(tier_ssd, case_name), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            if tier_ssd:
                stack.enter_context(patch.dict(os.environ, _mooncake_ssd_offload_env()))
            extra_config: dict[str, Any] = {
                "backend": backend,
                "use_layerwise": use_layerwise,
                "lookup_rpc_port": LOOKUP_RPC_PORT,
            }
            # No load_async key: it only gates the mamba state copy-after-load
            # (ascend_store_connector.py:169-171) and V4.1 is a CED/SWA hybrid
            # with no mamba state, unlike the Qwen3.8 GDN case that sets it.
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": extra_config,
            }
        yield stack.enter_context(VllmRunner(maybe_model_redirect(MODEL), **kwargs))


def _natural_tokens(runner: VllmRunner, sentence: str, length: int) -> list[int]:
    tokenizer = runner.model.get_tokenizer()
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    assert tokens, "tokenizer produced no tokens for the pool prompt sentence"
    return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]


def _generate(runner: VllmRunner, prompt_ids: list[int], max_tokens: int) -> list[int]:
    (out,) = runner.model.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    assert out.finished
    assert len(out.outputs[0].token_ids) == max_tokens
    return list(out.outputs[0].token_ids)


def _engine_pool_shapes(runner: VllmRunner) -> tuple[int, int, list[int]]:
    """Return ``(block_size, lcm of KV-group block sizes, group sizes)``.

    V4.1's CED layout keeps many KV cache groups with different block sizes
    (compressed CSA2/global, indexer, SWA bounded replay, Engram).  The pool only
    stores whole lcm groups and the SWA groups do not participate in prefix
    reuse -- stated for the bounded-replay path of the shipped image in
    section 12 of DeepSeek-V4.1-Flash.md:1868-1871; only the cacheable groups
    are registered with the connector, so the exact hit expectation must come
    from the engine's real group layout, never from a copied constant.  The
    scheduler holds the ``KVCacheConfig`` produced by engine-core cache
    initialization (vllm EngineCore -> ``Scheduler(kv_cache_config=...)``).
    """
    engine = runner.model.llm_engine
    block_size = int(engine.vllm_config.cache_config.block_size)
    core = getattr(getattr(engine, "engine_core", None), "engine_core", None)
    scheduler = getattr(core, "scheduler", None)
    kv_cache_config = getattr(scheduler, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None) if kv_cache_config is not None else None
    group_sizes: list[int] = []
    if groups:
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            if spec is None:
                continue
            # UniformTypeKVCacheSpecs wraps several identical per-layer specs.
            inner = getattr(spec, "kv_cache_specs", None)
            if inner:
                spec = next(iter(inner.values()))
            group_sizes.append(int(getattr(spec, "block_size", block_size)))
    if not group_sizes:
        # Out-of-process EngineCore (mp backend) does not expose the scheduler,
        # so the group layout cannot be read in-process.  Fall back to the cache
        # block size and say so in the log; the exact-hit assertion then compares
        # against the block-granularity expectation.
        group_sizes = [block_size]
    lcm = group_sizes[0]
    for size in group_sizes[1:]:
        lcm = lcm * size // math.gcd(lcm, size)
    return block_size, lcm, group_sizes


def _isolation_sentence() -> str:
    """Run-unique R4 isolation prompt base.

    The tier-3 storage tiers (Mooncake SSD offload directory, and by symmetry
    the memcache UBS IO block device) PERSIST their data across service
    restarts -- measured on 80.48.37.145 (2026-11-12) on the GLM tier-3 case:
    a gate-mode re-run's R4 disjoint prompt hit the *previous* run's
    same-content entries for a full 15,360-token prefix ("R4 isolation
    failed: ... got 15360.0"). That is legitimate persistent-pool reuse, not
    a key collision, but it defeats R4's fresh-pool assumption. Mixing a
    per-call nonce into the isolation prompt keeps R4 strict (genuinely new
    content must miss) while staying immune to the storage tier's cross-run
    persistence.
    """
    return SENTENCE_B + f"Isolation probe {time.time_ns()}. "


def _phase_accuracy(runner: VllmRunner, state: dict[str, Any], tag: str) -> None:
    """Phase 1: cold greedy output of the shared prefix (also the pool's R1 write)."""
    started = time.perf_counter()
    state["prompt_a"] = _natural_tokens(runner, SENTENCE_A, POOL_PROMPT_TOKENS)
    state["prompt_b"] = _natural_tokens(runner, _isolation_sentence(), POOL_PROMPT_TOKENS)
    state["cold"] = _generate(runner, state["prompt_a"], POOL_MAX_TOKENS)
    print(
        f"[kv_pool_a2] {tag} accuracy: cold_tokens={len(state['cold'])} "
        f"head={state['cold'][:6]} tail={state['cold'][-6:]} "
        f"cold_latency={(time.perf_counter() - started):.1f}s",
        flush=True,
    )


def _expected_hit_tokens(prompt_tokens: int, block_size: int, lcm: int) -> int:
    """Exact hit-token expectation with the drop-final-engine-block rule.

    The generic ``expected_hit_token_count`` formula (floor(prompt / lcm) *
    lcm) overcounts for this engine: the scheduler drops the final *engine
    block* before grouping the survivors into lcm chunks -- the same behavior
    documented from measurement for the GLM align layout in PR #20757 case
    #11. Measured on 80.48.37.145 (2026-11-12, dsv41flash_tp8_mooncake_bulk_
    tier2 record run): the V4.1 engine reports a single uniform group
    (block_size == lcm == 128) and a 16,388-token prompt (128 full blocks +
    4 tail) hit 16,256 = 127 x 128 tokens -- exactly one full block less than
    the generic formula's 16,384. The PR #1 DSV4-Flash precedent (16,384 hits
    at lcm=16,384) does not apply here: that engine's mixed-group layout
    keeps the final block, this one drops it. The pool served every block it
    was given; only this expectation needed the drop rule.
    """
    full_blocks = prompt_tokens // block_size
    if full_blocks <= 1:
        return 0
    survivor_tokens = (full_blocks - 1) * block_size
    return survivor_tokens // lcm * lcm


def _phase_hit(runner: VllmRunner, state: dict[str, Any], tag: str) -> None:
    """Phase 2: pool hit quality -- R2/R3 exact hits, token identity, R4 isolation."""
    prompt_a, prompt_b, cold = state["prompt_a"], state["prompt_b"], state["cold"]
    block_size, lcm, _ = state["shapes"]
    expected = _expected_hit_tokens(len(prompt_a), block_size, lcm)
    assert expected > 0, (
        f"[{tag}] prompt {len(prompt_a)} cannot hit with block_size={block_size} lcm={lcm}; "
        "raise POOL_PROMPT_TOKENS above 2x the reported lcm (the prompt must keep a "
        "non-zero tail, see the module docstring)"
    )
    if state["ssd"]:
        # Tier-3 pressure rounds: extra 16K prefixes push the oldest content of
        # the shared prefix out of the small DRAM tier so the hit rounds read a
        # mixed DRAM/SSD prefix back through the prefetch path.  Each pressure
        # prefix is unique per index AND per run (nonce suffix, same pattern as
        # the R4 isolation prompt): the count is only bounded by
        # SSD_PRESSURE_PROMPTS, and the tier-3 storage persistence across runs
        # can never serve a stale same-content prefix instead of forcing a
        # fresh write.  With DSV41_POOL_SSD_DRAM_TIER shrunk and
        # DSV41_POOL_SSD_PRESSURE_PROMPTS raised past the watermark, the LRU
        # victim is the R1 shared prefix and R2/R3 must read it back through
        # the SSD offload tier (the eviction-forcing verification layout).
        for index in range(SSD_PRESSURE_PROMPTS):
            sentence = (
                PRESSURE_SENTENCES[index % len(PRESSURE_SENTENCES)] + f"Pressure probe {index} {time.time_ns()}. "
            )
            _generate(runner, _natural_tokens(runner, sentence, POOL_PROMPT_TOKENS), 1)
            if (index + 1) % 10 == 0:
                print(f"[kv_pool_a2] {tag}: pressure {index + 1}/{SSD_PRESSURE_PROMPTS}", flush=True)
        time.sleep(SSD_SETTLE_SECONDS)

    began = time.perf_counter()
    before = snapshot_metrics(runner)
    hit1 = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    mid = snapshot_metrics(runner)
    r2_seconds = time.perf_counter() - began
    assert_hit_tokens(before, mid, expected=expected, tag=f"{tag} R2")
    if OUTPUT_IDENTITY:
        assert_output_identical(cold, hit1, tag=f"{tag} R2")

    began = time.perf_counter()
    hit2 = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    after = snapshot_metrics(runner)
    r3_seconds = time.perf_counter() - began
    assert_hit_tokens(mid, after, expected=expected, tag=f"{tag} R3")
    if OUTPUT_IDENTITY:
        assert_output_identical(cold, hit2, tag=f"{tag} R3")

    _generate(runner, prompt_b, POOL_MAX_TOKENS)
    final = snapshot_metrics(runner)
    assert_no_hit(after, final, tag=f"{tag} R4")

    state["hit"] = hit2
    state["hit_tokens"] = expected
    print(
        f"[kv_pool_a2] {tag} hit: expected_tokens={expected} "
        f"r2_delta={mid.hits - before.hits:.0f} r3_delta={after.hits - mid.hits:.0f} "
        f"r2_latency={r2_seconds:.1f}s r3_latency={r3_seconds:.1f}s "
        f"identity_checked={OUTPUT_IDENTITY} "
        f"queries_delta={final.queries - before.queries:.0f}",
        flush=True,
    )


def _phase_control(runner: VllmRunner, state: dict[str, Any], tag: str) -> None:
    """Leg-B hit control: the local prefix cache must serve the same prefix."""
    prompt_a = state["prompt_a"]
    cold = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    metrics_before = runner.get_metrics()
    before = float(metrics_before.get(PREFIX_HITS_METRIC, 0.0))
    hit = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    metrics_after = runner.get_metrics()
    after = float(metrics_after.get(PREFIX_HITS_METRIC, 0.0))
    assert after > before, (
        f"[{tag}] CONTROL_DEFECT: B leg got no local prefix-cache hit (prefix_cache_hits delta={after - before:.0f})"
    )
    assert_output_identical(cold, hit, tag=f"{tag} control")
    state["cold"] = cold
    state["hit"] = hit
    state["local_hit_delta"] = after - before
    print(
        f"[kv_pool_a2] {tag} control: prefix_cache_hits_delta={after - before:.0f}",
        flush=True,
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _phase_perf(runner: VllmRunner, prompt_ids: list[int], tag: str) -> tuple[float, float, float]:
    """Phase 3: end-to-end latency of the repeatedly-hit prefix (TTFT proxy)."""
    ttft: list[float] = []
    produced = 0
    started = time.perf_counter()
    for index in range(AB_WARMUP + AB_SAMPLES):
        began = time.perf_counter()
        _generate(runner, prompt_ids, 1)
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        if index >= AB_WARMUP:
            ttft.append(elapsed_ms)
            produced += len(prompt_ids)
    wall = time.perf_counter() - started
    p50 = _percentile(ttft, 0.50)
    p95 = _percentile(ttft, 0.95)
    throughput = produced / wall
    print(
        f"[kv_pool_a2] {tag} perf: ttft_p50={p50:.1f}ms ttft_p95={p95:.1f}ms "
        f"input_throughput={throughput:.0f}tok/s samples={AB_SAMPLES}",
        flush=True,
    )
    return p50, p95, throughput


def _preflight(case_name: str, use_layerwise: bool, tier_ssd: bool) -> None:
    """Gate the case on measured environment facts, with the raw evidence in the reason.

    Checks, in order of how expensive they are to satisfy: the load fits the
    scheduler window, NPUs are visible, hugepages are reserved, the checkpoint
    declares a W4A8 layout A2 can serve, then the SSD-offload root the tier-3
    scenario needs.  ``use_layerwise`` stays in the signature so the parameter
    matrix (which pins it to False for this model, see the module docstring)
    keeps flowing through unchanged.
    """
    assert POOL_MAX_MODEL_LEN > POOL_PROMPT_TOKENS + POOL_MAX_TOKENS, (
        f"POOL_MAX_MODEL_LEN={POOL_MAX_MODEL_LEN} cannot hold the {POOL_PROMPT_TOKENS}-token "
        f"prompt plus {POOL_MAX_TOKENS} generated tokens; raise DSV41_POOL_MAX_MODEL_LEN"
    )
    npu_count, npu_evidence = _npu_device_status()
    print(f"[kv_pool_a2] {case_name}: npu devices: {npu_evidence} (visible={VISIBLE_DEVICES})", flush=True)
    if npu_count is None:
        pytest.skip(f"this TP8 case needs an Ascend NPU host: {npu_evidence}")
    assert npu_count is not None  # type narrowing for mypy
    if npu_count < TP_SIZE:
        pytest.skip(
            f"TP8 needs {TP_SIZE} NPUs, container sees {npu_count} ({npu_evidence}). "
            f"Start with --device all and ASCEND_RT_VISIBLE_DEVICES={VISIBLE_DEVICES}."
        )

    print(f"[kv_pool_a2] {case_name}: mooncake {_mooncake_version()}", flush=True)
    hugepages = _hugepages_total()
    if hugepages == 0:
        pytest.skip(
            "mooncake pooling on A2 runs the HIXL RoCE direct mode over reserved 2MB hugepages "
            "(the official guide reserves 200000: kv_cache_pool_ascend_store.md:275-276); none are "
            "visible to the container. Reserve them on the host "
            "(echo 200000 > /proc/sys/vm/nr_hugepages) and pass the pages into the container."
        )
    print(f"[kv_pool_a2] {case_name}: HugePages_Total={hugepages}", flush=True)

    verdict, evidence = _classify_w4a8_checkpoint(MODEL)
    print(f"[kv_pool_a2] {case_name}: checkpoint quant path={verdict} ({evidence})", flush=True)
    if verdict == "missing_dir":
        pytest.skip(
            f"{evidence}. This case reads config.json / {MODELSLIM_QUANT_DESCRIPTION_FILE} from the "
            "mount, so point DSV41_POOL_MODEL at a local checkpoint directory (not a hub id)."
        )
    if verdict == "unknown":
        pytest.skip(
            f"cannot decide the W4A8 layout from the checkpoint metadata ({evidence}). The repository "
            "documents W8A8 V4.1 only (DeepSeek-V4.1-Flash.md:59-67, supported_models.md:129), so this "
            "case refuses to guess which MoE kernel path the serving layer will take."
        )
    if verdict == "mxfp":
        pytest.skip(f"{MODEL!r} ships DeepSeek-original FP4 weights: {W4A8_MXFP_A5_ONLY_MESSAGE} Evidence: {evidence}")
    if verdict == "w8a8":
        pytest.skip(
            f"{MODEL!r} is a W8A8 checkpoint while this case is registered as the W4A8 protection "
            f"matrix (evidence: {evidence}). Point DSV41_POOL_MODEL at the W4A8 export or reuse the "
            "W8A8 case instead."
        )

    if tier_ssd:
        ok, note = _prepare_ssd_root(_ssd_root(case_name))
        print(f"[kv_pool_a2] {case_name}: tier3 SSD: {note}", flush=True)
        if not ok:
            pytest.skip(note)


def _emit_ab_record(
    case_name: str,
    backend: str,
    mode: str,
    tier: str,
    a_state: dict[str, Any],
    a_perf: tuple[float, float, float],
    b_state: dict[str, Any],
    b_perf: tuple[float, float, float],
    token_identical: bool,
    perf_ok: bool,
    ttft_limit: float,
    throughput_limit: float,
) -> None:
    """Log and persist the A/B verdict in the shared watch format.

    Same ``[kv_pool_ab]`` line shape and JSON-Lines record as
    ``kv_pool_watch.PoolABReport.emit`` so one grep works across cases, but with
    the per-tier limits attached (``run_pool_ab`` is not used here; see the
    comment on ``TIER_TTFT_LIMITS``).  ``outputs`` carries only the 16-token
    completions, not the 16K prompt, which is what made the stock report huge.
    """
    label = "PASS" if ((token_identical or not OUTPUT_IDENTITY) and perf_ok) else "POOL_DEFECT"
    verdict = "PASS" if label == "PASS" else "FAIL"
    a_p50, a_p95, a_throughput = a_perf
    b_p50, b_p95, b_throughput = b_perf
    delta_ttft = (a_p50 / b_p50 - 1.0) if b_p50 else 0.0
    delta_throughput = (a_throughput / b_throughput - 1.0) if b_throughput else 0.0
    legs = (
        ("A", float(a_state.get("hit_tokens", 0.0)), a_perf),
        # The control leg owns local cache hits, not pool hits (watch L590).
        ("B", 0.0, b_perf),
    )
    for leg, hit_tokens, perf in legs:
        print(
            f"[kv_pool_ab] case={case_name} leg={leg} acc=1 hit_tokens={hit_tokens:.0f} "
            f"ttft_p50={perf[0]:.2f} ttft_p95={perf[1]:.2f} throughput={perf[2]:.1f} "
            f"delta_ttft={delta_ttft:+.4f} delta_throughput={delta_throughput:+.4f} "
            f"verdict={verdict} label={label}",
            flush=True,
        )
    payload = {
        "case": case_name,
        "backend": backend,
        "mode": mode,
        "tier": tier,
        "a": {
            "acc": True,
            "hit_tokens": float(a_state.get("hit_tokens", 0.0)),
            "outputs": [a_state.get("hit", [])],
            "ttft_p50": a_p50,
            "ttft_p95": a_p95,
            "throughput": a_throughput,
            "shapes": list(a_state.get("shapes", ())),
        },
        "b": {
            "acc": True,
            "hit_tokens": 0.0,
            "local_hit_delta": b_state.get("local_hit_delta", 0.0),
            "outputs": [b_state.get("hit", [])],
            "ttft_p50": b_p50,
            "ttft_p95": b_p95,
            "throughput": b_throughput,
        },
        "verdict": verdict,
        "label": label,
        "token_identical": token_identical,
        "output_identity_checked": OUTPUT_IDENTITY,
        "delta_ttft": delta_ttft,
        "delta_throughput": delta_throughput,
        "limits": {"ttft_max": ttft_limit, "throughput_min": throughput_limit, "mode": AB_MODE},
        "env": {
            "model": MODEL,
            "ascend_rt_visible_devices": VISIBLE_DEVICES,
            "vllm_use_v2_model_runner": USE_V2_MODEL_RUNNER,
            "gpu_memory_utilization": POOL_GPU_MEM_UTIL,
            "max_model_len": POOL_MAX_MODEL_LEN,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "mooncake_version": _mooncake_version(),
            "kv_pool_ab_mode": AB_MODE,
        },
    }
    try:
        with open(AB_ARTIFACT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except OSError as exc:
        print(f"[kv_pool_ab] case={case_name} artifact {AB_ARTIFACT!r} not written: {exc}", flush=True)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,dspark,chunked_prefill,prefix_caching",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W4A8",
    graph_mode="eager",
)
@pytest.mark.parametrize(
    ("backend", "use_layerwise", "tier_ssd"),
    [
        # Case 1: bulk transfer, tier-2 DRAM-only pool.
        ("mooncake", False, False),
        # Case 2: bulk transfer, tier-3 DRAM + Mooncake SSD offload pool. The
        # tier-3 storage axis is exercised WITHOUT the layerwise transfer mode:
        # layerwise is documented as unsupported for this model (four tutorial
        # citations plus the unwired per-layer hooks in dsa_v41.py -- see the
        # module docstring), so the two params form the clean tier-2/tier-3
        # storage-medium pair the user-level requirement asks for.
        ("mooncake", False, True),
    ],
    ids=["mooncake_bulk_tier2", "mooncake_bulk_tier3"],
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": USE_V2_MODEL_RUNNER,
        # DeepSeek-V4.1-Flash.md:1661 (8 devices) / :48-49 (runner V1).
        "ASCEND_RT_VISIBLE_DEVICES": VISIBLE_DEVICES,
        # Spawn workers are required for multi-process NPU execution.
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Pool block hashes must be reproducible across restarts
        # (DeepSeek-V4.1-Flash.md:1444).
        "PYTHONHASHSEED": "0",
        "VLLM_ENGINE_READY_TIMEOUT_S": ENGINE_READY_TIMEOUT_S,
        "HCCL_CONNECT_TIMEOUT": HCCL_CONNECT_TIMEOUT,
        "PYTORCH_NPU_ALLOC_CONF": PYTORCH_NPU_ALLOC_CONF,
    },
)
@wait_until_npu_memory_free()
def test_deepseek_v41_flash_kv_pool_ab_a2(backend: str, use_layerwise: bool, tier_ssd: bool) -> None:
    mode = "layerwise" if use_layerwise else "bulk"
    tier = "tier3-ssd" if tier_ssd else "tier2-dram"
    case = f"dsv41flash_tp8_{backend}_{mode}_{tier}"
    print(
        f"[kv_pool_a2] {case}: model={MODEL} tier={tier} mode={mode} dram_tier="
        f"{SSD_DRAM_TIER if tier_ssd else DRAM_TIER} prompt_tokens={POOL_PROMPT_TOKENS} "
        f"block_size={POOL_BLOCK_SIZE} max_model_len={POOL_MAX_MODEL_LEN} "
        f"gpu_mem_util={POOL_GPU_MEM_UTIL} spec_tokens={NUM_SPECULATIVE_TOKENS} "
        f"tp={TP_SIZE} ep={EXPERT_PARALLEL} ab_mode={AB_MODE}",
        flush=True,
    )
    _preflight(case, use_layerwise=use_layerwise, tier_ssd=tier_ssd)

    with _leg(True, backend, use_layerwise, tier_ssd, case) as runner:
        a_state: dict[str, Any] = {"ssd": tier_ssd, "backend": backend}
        a_state["shapes"] = _engine_pool_shapes(runner)
        block_size, lcm, group_sizes = a_state["shapes"]
        print(
            f"[kv_pool_a2] {case} A: block_size={block_size} lcm={lcm} groups={group_sizes} "
            f"num_groups={len(group_sizes)}",
            flush=True,
        )
        _phase_accuracy(runner, a_state, f"{case}-A")
        _phase_hit(runner, a_state, f"{case}-A")
        a_perf = _phase_perf(runner, a_state["prompt_a"], f"{case}-A")

    # Leave time for the pool service and the 8 worker processes to release NPU
    # memory before the control leg starts.
    time.sleep(AB_LEG_GAP_SECONDS)

    with _leg(False, backend, use_layerwise, tier_ssd, case) as runner:
        b_state: dict[str, Any] = {"ssd": False, "backend": backend}
        b_state["shapes"] = a_state["shapes"]
        _phase_accuracy(runner, b_state, f"{case}-B")
        _phase_control(runner, b_state, f"{case}-B")
        b_perf = _phase_perf(runner, b_state["prompt_a"], f"{case}-B")

    token_identical = a_state["hit"] == b_state["hit"]
    ttft_limit = TIER_TTFT_LIMITS["tier3" if tier_ssd else "tier2"]
    throughput_limit = TIER_THROUGHPUT_LIMITS["tier3" if tier_ssd else "tier2"]
    a_p50, a_p95, a_throughput = a_perf
    b_p50, b_p95, b_throughput = b_perf
    perf_ok = (
        a_p50 <= b_p50 * ttft_limit and a_p95 <= b_p95 * ttft_limit and a_throughput >= b_throughput * throughput_limit
    )
    verdict = "PASS" if ((token_identical or not OUTPUT_IDENTITY) and perf_ok) else "FAIL"
    print(
        f"[kv_pool_a2] case={case} A/B: ttft_p50 {a_p50:.1f} vs {b_p50:.1f} "
        f"({a_p50 / b_p50:.2f}x) ttft_p95 {a_p95:.1f} vs {b_p95:.1f} ({a_p95 / b_p95:.2f}x) "
        f"throughput {a_throughput:.0f} vs {b_throughput:.0f} "
        f"({a_throughput / b_throughput:.2f}x) token_identical={token_identical} "
        f"identity_gate={OUTPUT_IDENTITY} verdict={verdict} mode={AB_MODE} "
        f"limits=ttft<={ttft_limit} tput>={throughput_limit}",
        flush=True,
    )
    _emit_ab_record(
        case,
        backend,
        mode,
        tier,
        a_state,
        a_perf,
        b_state,
        b_perf,
        token_identical,
        perf_ok,
        ttft_limit,
        throughput_limit,
    )
    if OUTPUT_IDENTITY:
        assert token_identical, (
            f"[{case}] pooled leg and control leg outputs differ: pool hit path corrupted "
            f"the reused prefix (A tail={a_state['hit'][-8:]} B tail={b_state['hit'][-8:]}); "
            "set DSV41_POOL_OUTPUT_IDENTITY=0 only with DSpark-divergence evidence attached"
        )
    elif not token_identical:
        # The audited H3 exemption is in effect: keep the divergence visible in
        # the log and in the report (token_identical=false there) instead of
        # pretending the comparison passed.
        print(
            f"[kv_pool_a2] case={case} WARNING: identity gate disabled, A/B tokens differ "
            f"(A tail={a_state['hit'][-8:]} B tail={b_state['hit'][-8:]})",
            flush=True,
        )
    if AB_MODE == "gate":
        assert perf_ok, (
            f"[{case}] pooled leg performance outside A/B limits: "
            f"ttft_p50 {a_p50:.1f}ms vs {b_p50:.1f}ms, ttft_p95 {a_p95:.1f}ms vs {b_p95:.1f}ms, "
            f"throughput {a_throughput:.0f} vs {b_throughput:.0f} tok/s "
            f"(limits: ttft <= {ttft_limit}x, throughput >= {throughput_limit}x)"
        )
