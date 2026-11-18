# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A2 single-node KV-pool A/B protection cases for GLM-5.3-Flash W8A8 (TP8) x memcache.

Two scenarios guard the A2 TP8 pooling matrix for the hybrid-KDA GLM-5.3-Flash
checkpoint (the A3 four-card pair and the A2 two-card Qwen3.8 pair already
cover the other cells; this file is the A2/TP8 memcache column):

========================  =========  ==========  ==========
scenario                  backend    mode        tier (medium)
========================  =========  ==========  ==========
memcache_bulk_tier2       memcache   bulk        DRAM
memcache_layerwise_tier3  memcache   layerwise   DRAM + SSD
========================  =========  ==========  ==========

Two orthogonal axes, not one: ``bulk`` vs ``layerwise`` is the *transfer mode*
axis (whole-request synchronous load vs per-layer streamed load, connector
extra_config ``use_layerwise``); ``tier2`` vs ``tier3`` is the *storage medium*
axis (pool DRAM only vs pool DRAM spilling to a UBS IO raw block device).
This pair happens to correlate them (tier2 runs bulk, tier3 runs layerwise,
mirroring the Qwen3.8 A2 pair) but they are configured independently and the
case names keep both axes visible so a reader never conflates medium with
transfer mode.

Every scenario runs two control legs, each inside one engine instance:

- Leg A: pool enabled (memcache, per-scenario transfer mode / tier), local
  prefix cache **disabled**;
- Leg B: pool disabled, local prefix cache enabled.

Both legs run three phases inside the single service: accuracy (cold greedy
output of a shared 16K natural-language prefix), pool hit (R2/R3 exact
hit-token count against ``expected_hit_token_count``, token identity against
the cold output, R4 disjoint-prompt miss), and performance (TTFT-proxy
sampling on the repeatedly-hit prefix). The A/B comparison asserts the pooled
leg reproduces the control leg's tokens exactly and stays within the per-tier
performance gates (``TIER_TTFT_LIMITS`` / ``TIER_THROUGHPUT_LIMITS``,
calibrated on the A2 Qwen3.8 pair and env-overridable for GLM-specific
recalibration).

Hybrid align deviation (documented, never silent): GLM-5.3-Flash is a hybrid
KDA linear-attention model and ``AscendStoreConnector`` refuses any other
mamba cache mode (``ascend_store/pool_scheduler.py:223-226`` requires
``mamba_cache_mode="align"``). The deployment docs bind ``align`` to the local
prefix cache being enabled (``GLM5.3-Flash.md:1464-1470``: "the model
configuration selects this mode when prefix caching is enabled"; the A3 pair
#10/#11 keep the local cache on in *both* legs). The user-level A/B definition
asks for leg A = pool on + local cache **off**, so this file implements leg A
in strict mode first; if (and only if) the engine rejects that combination at
startup, the leg is rebuilt in fallback mode: both legs keep ``align`` + local
prefix cache ON, leg A calls ``reset_prefix_cache(reset_connector=False)``
before every round/sample so the pool stays the only possible hit source, and
a ``[kv_pool_a2] deviation ...`` line plus the ``ab_local_cache`` /
``deviation_reason`` fields of the JSON artifact record which mode ran and
why. The judged criteria (exact external hit tokens, R4 isolation, token
identity) are identical in both modes; nothing is relaxed silently.

Case ④ (``memcache_layerwise_tier3``) is the evidence-strongest cell here:
the v0.30.0 release notes state "GLM-5.3-Flash: ... added layerwise memcache
DRAM/SSD pooling" (``docs/source/user_guide/release_notes.md:15``,
:143 ``#17698``). The GLM5.3-Flash tutorial's A2 pool paragraph ("Keep ...
`use_layerwise` disabled for this configuration", :1681-1682, and the A3
:1470 same-worded note) describes the *validated deployment configuration*,
not a capability prohibition — the docs table at :25-26 similarly restricts
the model runner only. Both citations are kept side by side deliberately so a
reviewer can see the case is asserting a shipped feature, not a hoped-for one.
``KV_POOL_AB_MODE=record`` on the very first production run remains the safe
rollout choice; the per-tier gates are overridable without editing this file.

Sizing risk (over documented configuration, read before triaging OOM): the
320B-parameter W8A8 checkpoint is ~320GB of weights, and the documented A2
baseline for this checkpoint is **2 nodes colocated** (DP2 x TP8,
``GLM5.3-Flash.md:36-38``) with a 200k-class context matrix. This watch case
deliberately squeezes it into one 8x64GB node with a 16K-class window
(``gpu_memory_utilization`` default 0.85, ``max_model_len`` 16640) to exercise
the pool, not the context length. If engine startup fails from HBM pressure
rather than a pool defect, retune ``GLM53_POOL_GPU_MEM_UTIL`` /
``GLM53_POOL_MAX_MODEL_LEN`` (or move the case to TP8 x DP2 hardware) before
suspecting the connector.

Model-runner choice: ``VLLM_USE_V2_MODEL_RUNNER=0`` per
``GLM5.3-Flash.md:25-26`` ("GLM-5.3-Flash model currently supports only model
runner V1 on Ascend"). The Qwen3.8 A2 precedent runs V2 (=1); the difference
is model-specific and intentional, not drift.

Environment knobs:

- ``GLM53_POOL_MODEL``: local checkpoint dir (default
  ``/mnt/share/weight/GLM-5.3-Flash-w8a8-requant-20260917``). The plain
  ``GLM-5.3-Flash-w8a8`` export ships 80 weight shards but **no tokenizer
  files at all** (no tokenizer.json / tokenizer_config.json / vocab /
  merges, and config.json carries ``tokenizer_class=null``), so vLLM cannot
  even build its tokenizer group from it; the requant sibling (82 shards,
  same model config) is the complete, runnable W8A8 checkpoint and is the
  selected default. Point this env at the plain dir only after its tokenizer
  files are provided;
- ``GLM53_POOL_GPU_MEM_UTIL`` (default 0.85) / ``GLM53_POOL_MAX_MODEL_LEN``
  (default 16640): HBM retuning knobs for the over-doc single-node layout;
- ``GLM53_POOL_NUM_SPEC_TOKENS``: MTP speculative tokens (default 5, the
  documented A2 pool example ``GLM5.3-Flash.md:1753/1780`` uses 5);
- ``KV_POOL_GLM53_TIER2_CUDAGRAPH_MODE``: tier2 graph mode (default
  ``FULL_DECODE_ONLY``; ``eager`` is the documented alternative). Tier3 is
  forced to ``PIECEWISE`` because ``use_layerwise`` requires it
  (``ascend_store_connector.py:118-124``);
- ``KV_POOL_GLM53_DRAM_TIER_GB`` (default 4) / ``KV_POOL_GLM53_SSD_DRAM_TIER_GB``
  (default 3): per-rank memcache pool DRAM sizes (device_rdma auto-aligns to
  1GB, guide :806);
- ``KV_POOL_GLM53_PROMPT_MB_PER_RANK`` (default 600): planning estimate of one
  16K prompt's pool payload per rank, used only to derive the eviction
  watermarks and print them (see the derivation next to the constants);
- ``KV_POOL_GLM53_EVICT_HIGH`` / ``KV_POOL_GLM53_EVICT_LOW``: override the
  derived ~80/20 DRAM/SSD split watermarks;
- ``KV_POOL_GLM53_UBSIO_MEM_GB``: ``ubsio.mem.size_in_gb`` per die (default 10);
- ``KV_POOL_MEMCACHE_UBSIO_DISK``: raw block device for the tier3 UBS IO
  storage (default ``/dev/loop0`` as a placeholder; on the validated A2
  runner the environment provisions dedicated ``/dev/loopN`` devices and this
  env **must** be pointed at the reserved one);
- ``KV_POOL_MEMCACHE_UBSIO_DEVICES``: ``ubsio.standalone.device_count``
  (default ``TP_SIZE`` = 8: one DRAM-enabled LocalService per worker, guide
  :811 "8 on A2");
- ``KV_POOL_GLM53_TTFT_LIMIT_TIER2/_TIER3`` / ``KV_POOL_GLM53_TPUT_LIMIT_TIER2/_TIER3``:
  per-tier A/B performance gates;
- ``KV_POOL_AB_MODE``: ``gate`` (default, A/B comparison must pass — same
  convention as the Qwen3.8 A2 file) or ``record`` (log + artifact only; the
  shared ``run_pool_ab`` default is ``record``, but this file does not call
  ``run_pool_ab`` because its thresholds are module constants);
- ``KV_POOL_AB_ARTIFACT``: JSONL artifact path, honored from ``kv_pool_watch``.

Host / runner prerequisites: an A2-class machine with 8 NPUs visible, the
``memcache_hybrid`` package importable, a complete checkpoint dir, and (for
tier3) a dedicated raw UBS IO block device (unmounted, no filesystem
signature; a loop device is the documented option, guide :809). The
``device_rdma`` protocol needs a current A2 driver stack — on the validated
runner (80.48.37.148, driver 25.5.6, image af76d4559efe, memcache_hybrid
1.3.0) it comes up; an older driver fails the pool bootstrap
(``WaitQpReady`` / ``hybm_transport_connect`` error -7), which reads as a pool
defect but is an environment gap, so the case skips, never guesses.
"""

import json
import math
import os
import time
from contextlib import ExitStack, contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
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

# Default checkpoint: the requant dir is the complete W8A8 export (82/82
# shards plus the full tokenizer set). The plain GLM-5.3-Flash-w8a8 dir has
# 80/80 weight shards but no tokenizer files (measured on the A2 runner:
# tokenizer.json/tokenizer_config.json/vocab/merges all absent,
# config.json tokenizer_class=null), which makes engine startup fail before
# any pool logic runs; it stays reachable via GLM53_POOL_MODEL only for
# environments that provide the tokenizer alongside it.
MODEL = os.environ.get("GLM53_POOL_MODEL", "Eco-Tech/GLM-5.3-Flash-w8a8")
TP_SIZE = 8

# 16K-class shared prefix with a non-zero tail: the pool only transfers whole
# lcm groups, and an lcm-aligned prompt would trim its final block into a
# guaranteed miss (the tail-trim half of the rationale lives in
# ``kv_pool_watch.expected_hit_token_count``; this engine additionally drops
# the final *engine block* before lcm grouping — see
# ``_expected_hit_tokens``). The GLM-5.3 dummy four-card case measured
# lcm=1152 -> 16,128 hit tokens at prompt 16,388 (PR case #11); the real TP8
# engine reports block=lcm=640 and hits 15,360 = 24 of 25 full blocks. Both
# anchors are *cross-checks for the log line only* — the assertion always
# uses the values read from the live engine.
DUMMY_REF_LCM = 1152
DUMMY_REF_HITS = 16128
REAL_REF_HITS_TP8 = 15360
POOL_PROMPT_TOKENS = 16388
# Fix the chunked-prefill shape for cold and hit rounds alike; 16,388 tokens
# prefill in [16384, 4] chunks in every leg.
POOL_MAX_NUM_BATCHED_TOKENS = 16384
POOL_MAX_TOKENS = 16
POOL_GPU_MEM_UTIL = float(os.environ.get("GLM53_POOL_GPU_MEM_UTIL", "0.85"))
POOL_MAX_MODEL_LEN = int(os.environ.get("GLM53_POOL_MAX_MODEL_LEN", "16640"))
POOL_MIN_MAX_MODEL_LEN = POOL_PROMPT_TOKENS + POOL_MAX_TOKENS
POOL_NUM_SPEC_TOKENS = int(os.environ.get("GLM53_POOL_NUM_SPEC_TOKENS", "5"))

# The watch is text-only; the multimodal wrapper is still built (the
# production Glm5Next wrapper initializes its vision tower first), so mm
# input is limited to zero instead of pretending the tower is absent.
LIMIT_MM_TEXT_ONLY = {"image": 0, "video": 0}

TIER2_CUDAGRAPH_MODE = os.environ.get("KV_POOL_GLM53_TIER2_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
# use_layerwise hard-requires PIECEWISE graphs (ascend_store_connector.py
# :118-124 requires_piecewise_for_cudagraph); the GLM support matrix lists
# Piecewise as validated for this model, so tier3 is not configurable here.
TIER3_CUDAGRAPH_MODE = "PIECEWISE"

# memcache pool DRAM per rank (``dram.size`` is per die; device_rdma
# auto-aligns it to 1GB, guide :806, so both tiers stay on the 1GB grid).
DRAM_TIER_GB = int(os.environ.get("KV_POOL_GLM53_DRAM_TIER_GB", "4"))
SSD_DRAM_TIER_GB = int(os.environ.get("KV_POOL_GLM53_SSD_DRAM_TIER_GB", "3"))
DRAM_TIER = f"{DRAM_TIER_GB}GB"
SSD_DRAM_TIER = f"{SSD_DRAM_TIER_GB}GB"
# memcache LocalService cluster slots (guide :804, range [1,1024]); the pool
# serves one engine at a time here, 256 matches every existing watch case.
MEMCACHE_LOCAL_WORLD_SIZE = 256

AB_MODE = os.environ.get("KV_POOL_AB_MODE", "gate")
MEMCACHE_UBSIO_DISK = os.environ.get("KV_POOL_MEMCACHE_UBSIO_DISK", "/dev/loop0")
MEMCACHE_UBSIO_DEVICES = int(os.environ.get("KV_POOL_MEMCACHE_UBSIO_DEVICES", TP_SIZE))
MEMCACHE_UBSIO_MEM_GB = int(os.environ.get("KV_POOL_GLM53_UBSIO_MEM_GB", "10"))


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


# Performance gates are per storage tier. Calibrated on the A2 Qwen3.8 pair
# (same machine class, same memcache backends): a pool hit transfers ~GBs per
# request and costs 5-25% TTFT / 4-17% throughput at tier2; the tier3
# layerwise path streams per-layer loads and was measured at 0.42-0.46x
# throughput, so it only gates catastrophic regressions (a full re-prefill
# miss runs ~12x slower). GLM TP8 moves more bytes per rank than Qwen TP2 at
# the same window, hence every limit is env-overridable for recalibration
# without touching the file.
TIER_TTFT_LIMITS = {
    "tier2": _float_env("KV_POOL_GLM53_TTFT_LIMIT_TIER2", 1.35),
    "tier3": _float_env("KV_POOL_GLM53_TTFT_LIMIT_TIER3", 3.0),
}
TIER_THROUGHPUT_LIMITS = {
    "tier2": _float_env("KV_POOL_GLM53_TPUT_LIMIT_TIER2", 0.75),
    "tier3": _float_env("KV_POOL_GLM53_TPUT_LIMIT_TIER3", 0.30),
}

# --- Tier-3 ~80/20 DRAM/SSD split: sizing derivation ---------------------------------
# The hit phase writes, in order: the shared prefix (R1), then
# SSD_PRESSURE_PROMPTS fresh 16K prefixes as pressure, then settles and reads
# the *shared* prefix back. Everything written before the pressure prefixes is
# the LRU victim set, so eviction must (a) not trigger while only shared +
# first pressure prefix are resident, (b) trigger near the end of the second
# pressure prefix, and (c) stop low enough to leave DRAM for the rewarm
# readback. Planning payload of one 16K prompt per rank: the pool stores
# floor(16,388 / 1152) * 1152 = 16,128 tokens (the dummy-weights anchor; the
# live lcm is asserted separately), and the hybrid per-token payload is
# dominated by the DSA layers (MLA latent kv_lora_rank=512 x bf16 plus the
# 32-head x 128-dim indexer keys, guide widths in the four-card asset config)
# while the KDA groups only add align snapshots once per mamba block. That
# lands at hundreds of MB per rank: the planning constant keeps the Qwen3.8
# A2 measured anchor (600 MB/rank at TP2) and is env-overridable; recalibrate
# it from the first run's ``dram_used`` log, the watermark formulas below
# re-derive automatically.
PROMPT_POOL_MB_PER_RANK = int(os.environ.get("KV_POOL_GLM53_PROMPT_MB_PER_RANK", "600"))
SSD_PRESSURE_PROMPTS = 2
SSD_SETTLE_SECONDS = 30
# Target share of the *shared prefix* that should end up tiered down to SSD:
# production-realistic reads are ~80% fast-tier / ~20% SSD.
TARGET_SPILL_FRACTION = 0.20
_WORKING_SET_MB = PROMPT_POOL_MB_PER_RANK * (1 + SSD_PRESSURE_PROMPTS)
_SSD_TIER_MB = SSD_DRAM_TIER_GB * 1024
# Trigger watermark: just below the full working-set utilization, so the
# crossing happens during the last pressure prefix only. (Defaults:
# working set 1800MB of 3072MB -> high=58; shared alone 19.5%, shared+one
# pressure 39.1%, both stay far below 58 -> no premature spill.)
MEMCACHE_EVICT_THRESHOLD_HIGH = int(os.environ.get("KV_POOL_GLM53_EVICT_HIGH", _WORKING_SET_MB * 100 // _SSD_TIER_MB))
# Stop watermark: the (high - low) band of the DRAM tier is what gets spilled,
# so choose the band width to free ~TARGET_SPILL_FRACTION of one prefix.
# (Defaults: 0.2 * 600MB of 3072MB -> 4 percentage points -> low=54, spill
# ~=123-141MB ~= 20-23% of the shared prefix; the remaining ~77-80% is served
# from DRAM and 46% of the tier stays free for the rewarm readback.)
MEMCACHE_EVICT_THRESHOLD_LOW = int(
    os.environ.get(
        "KV_POOL_GLM53_EVICT_LOW",
        MEMCACHE_EVICT_THRESHOLD_HIGH
        - max(1, round(TARGET_SPILL_FRACTION * PROMPT_POOL_MB_PER_RANK * 100 / _SSD_TIER_MB)),
    )
)
# --- End tier-3 sizing derivation -------------------------------------------------------

# Leg-A local-cache policy modes for the hybrid align deviation (see module
# docstring): strict = user-defined A (pool on, local cache off); fallback =
# both legs keep align + local cache, leg A resets the local KV per round.
MODE_STRICT = "strict_a_local_off"
MODE_ALIGN_FALLBACK = "align_fallback_both_local_on"

# Natural-language prompts: random token ids flip under batch-shape
# differences even with intact KV, so H3/accuracy must use low-entropy text.
SENTENCE_A = "Explain how historical records help people understand the past. "
SENTENCE_B = "The mountain railway climbs steadily toward the old observatory. "
PRESSURE_SENTENCES = [
    "Draft a travel plan that lists every museum worth visiting in the capital. ",
    "Summarize the key milestones in the history of long-distance communication. ",
    "Describe the process of turning raw wool into warm winter clothing. ",
    "Outline the main arguments for and against daylight saving time. ",
]


class LegStartupError(RuntimeError):
    """The engine refused to start; no phase ran, so a retry is safe.

    Raised only around ``VllmRunner`` construction. Assertion failures from
    the phases propagate as plain ``AssertionError`` and never trigger the
    fallback rebuild.
    """


def _require_a2_devices() -> None:
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        pytest.skip("GLM-5.3-Flash A2 TP8 pool cases need Ascend NPUs (torch.npu), found none in this container")
    assert npu is not None  # type narrowing for mypy
    if npu.device_count() < TP_SIZE:
        pytest.skip(
            f"case needs {TP_SIZE} visible NPUs, container exposes {npu.device_count()} "
            "(check ASCEND_RT_VISIBLE_DEVICES)"
        )


def _require_checkpoint() -> None:
    root = Path(MODEL)
    if not root.is_dir():
        pytest.skip(
            f"checkpoint dir {MODEL!r} not found; mount the w8a8 weights or point GLM53_POOL_MODEL at "
            f"them (default: GLM-5.3-Flash-w8a8-requant-20260917; the plain "
            "GLM-5.3-Flash-w8a8 dir has 80 shards but no tokenizer files)"
        )
    if not (root / "config.json").is_file():
        pytest.skip(f"checkpoint {MODEL!r} has no config.json")
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        pytest.skip(f"checkpoint {MODEL!r} contains no .safetensors shards")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        expected_shards = int(json.loads(index.read_text(encoding="utf-8"))["metadata"]["total_shard_count"])
        if len(shards) != expected_shards:
            pytest.skip(f"checkpoint {MODEL!r} incomplete: {len(shards)}/{expected_shards} safetensors shards present")


def _require_memcache() -> None:
    if find_spec("memcache_hybrid") is None:
        pytest.skip(
            "memcache backend package 'memcache_hybrid' is not importable; install the KV pool "
            "backend per docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md "
            "(Scenario 2: Memcache Backend) before running these cases"
        )


def _require_ubsio_disk() -> None:
    if not os.path.exists(MEMCACHE_UBSIO_DISK):
        pytest.skip(
            f"memcache tier-3 needs the raw UBS IO block device {MEMCACHE_UBSIO_DISK!r} "
            "(dedicated, no mount point, no filesystem signature; a loop device is fine - provision "
            "one on the host, e.g. truncate -s 64G file && losetup -f --show file, pass it into the "
            "container with --device and set KV_POOL_MEMCACHE_UBSIO_DISK; the validated A2 runner "
            "provisions /dev/loopN devices per run, the default /dev/loop0 is only a placeholder)"
        )


def _memcache_pool_config(tier_ssd: bool) -> MemcacheKVPoolConfig:
    local: dict[str, Any] = {
        "ock.mmc.log_level": "error",
        "ock.mmc.local_service.world_size": MEMCACHE_LOCAL_WORLD_SIZE,
        # Official A2 recommendation (KV pool guide :805): device_rdma (RoCE)
        # for both tiers; dram.size stays on the 1GB alignment grid that
        # device_rdma forces (guide :806), so no host_shm fallback is needed
        # for the watermark math above.
        "ock.mmc.local_service.protocol": "device_rdma",
        "ock.mmc.local_service.dram.size": SSD_DRAM_TIER if tier_ssd else DRAM_TIER,
    }
    meta: dict[str, Any] = {
        "ock.mmc.log_level": "error",
        # The meta service also starts an HTTP metrics endpoint. Pin it to a
        # free port: the default 8000 collides with host services under
        # --network=host.
        "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
    }
    if tier_ssd:
        # Tier 3 with the ~80/20 DRAM/SSD split. The evict thresholds live in
        # the meta-service config (guide :774-775; defaults 90/80, the guide
        # suggests 70/60 "to improve SSD cache hit rate" when SSD is enabled -
        # the watermarks derived above are tighter than that recommendation
        # on purpose: the watch wants a *controlled partial* spill of ~20% of
        # one prefix, not aggressive SSD promotion, and they stay inside the
        # documented [1,99]/[1,98] ranges).
        meta["ock.mmc.evict_threshold_high"] = MEMCACHE_EVICT_THRESHOLD_HIGH
        meta["ock.mmc.evict_threshold_low"] = MEMCACHE_EVICT_THRESHOLD_LOW
        # UBS IO storage tier (guide :789-812): raw device, per-die cache,
        # one standalone device per DRAM-enabled LocalService (8 on A2 TP8).
        local["ock.mmc.local_service.storage.enabled"] = True
        local["ubsio.disk.path"] = MEMCACHE_UBSIO_DISK
        local["ubsio.mem.size_in_gb"] = MEMCACHE_UBSIO_MEM_GB
        local["ubsio.standalone.device_count"] = MEMCACHE_UBSIO_DEVICES
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={"meta": meta, "local": local},
    )


@contextmanager
def _leg(use_pool: bool, use_layerwise: bool, tier_ssd: bool, local_cache: bool, case_name: str):
    """Build one leg; the pool and the local prefix cache are mutually the control.

    - Leg A strict: ``use_pool=True, local_cache=False`` (user-level A/B).
    - Leg A fallback (hybrid align deviation): ``use_pool=True, local_cache=True``
      plus per-round ``reset_prefix_cache(reset_connector=False)`` in the phases.
    - Leg B: ``use_pool=False, local_cache=True``.

    Everything else (graph mode per scenario, parallel layout, scheduler
    budget, MTP spec) stays identical across the legs so the only variable is
    where the reused prefix comes from: the pool or the local prefix cache.
    """
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            dtype="auto",
            # The checkpoint config.json carries no quantization_config (the
            # modelslim output is described by quant_model_description.json),
            # so the Ascend quantizer must be selected explicitly.
            quantization="ascend",
            tensor_parallel_size=TP_SIZE,
            # MoE layout: the model docs call --enable-expert-parallel a must
            # for GLM-5.3-Flash (GLM5.3-Flash.md:313).
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=POOL_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=POOL_GPU_MEM_UTIL,
            seed=1024,
            enable_prefix_caching=local_cache,
            enable_chunked_prefill=True,
            disable_log_stats=False,
            # Hybrid KDA state: pool_scheduler.py:223-226 refuses any other
            # mamba cache mode for AscendStore. HMA (hybrid KV cache manager)
            # must stay enabled - glm5next/cache_config.py:382 raises if it is
            # disabled; there is no knob to turn it off here, which is the
            # safe direction.
            mamba_cache_mode="align",
            limit_mm_per_prompt=LIMIT_MM_TEXT_ONLY,
            # MTP speculative decoding per the documented A2 pool example
            # (method deepseek_mtp, 5 draft tokens, draft graph enabled).
            speculative_config={"method": "deepseek_mtp", "num_speculative_tokens": POOL_NUM_SPEC_TOKENS},
            compilation_config={
                "cudagraph_mode": TIER3_CUDAGRAPH_MODE if use_layerwise else TIER2_CUDAGRAPH_MODE,
            },
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_memcache_pool_config(tier_ssd), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            extra_config: dict[str, Any] = {
                "backend": "memcache",
                "use_layerwise": use_layerwise,
                "lookup_rpc_port": "0",
            }
            if not use_layerwise:
                # Hybrid bulk path MUST stay synchronous: the connector only
                # copies the mamba state after a load when
                # ``not use_layerwise and not load_async``
                # (ascend_store_connector.py:171); leaving load_async at its
                # default would silently serve hits with a stale recurrent
                # state (PR #10 notes: same requirement, same connector).
                # The layerwise scheduler hard-sets load_async=False itself
                # (layerwise/pool_scheduler.py:52), so the flag is bulk-only.
                extra_config["load_async"] = False
            kwargs["kv_transfer_config"] = {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_both",
                # Hybrid models must never fall back to recompute on a load
                # failure (platform.py:1310-1312 raises for recompute on
                # hybrid); fail-fast turns corruption into a loud error.
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": extra_config,
            }
        try:
            runner = stack.enter_context(VllmRunner(maybe_model_redirect(MODEL), **kwargs))
        except Exception as exc:  # noqa: BLE001 - startup failures carry unknown types;
            # only LegStartupError callers decide strict-vs-fallback, phases never retry.
            raise LegStartupError(f"[{case_name}] {type(exc).__name__}: {exc}") from exc
        yield runner


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


def _engine_pool_shapes(runner: VllmRunner) -> tuple[int, int]:
    """Return ``(block_size, lcm of KV-group block sizes)`` from the engine.

    GLM-5.3-Flash keeps several KV cache groups with different block sizes
    (MLA latent + DSA indexer keys + KDA align state); the pool only stores
    whole lcm groups, so the exact hit-token expectation must come from the
    engine's real group layout. The scheduler holds the ``KVCacheConfig``
    produced by the engine core's cache initialization (see vllm EngineCore ->
    Scheduler(kv_cache_config=...)). Dummy-weight reference (PR case #11,
    TP4): block_size=128, lcm=1152 -> expected hits 16,128 at prompt 16,388;
    that pair is printed as a cross-check but never asserted.
    """
    engine = runner.model.llm_engine
    block_size = int(engine.vllm_config.cache_config.block_size)
    core = getattr(getattr(engine, "engine_core", None), "engine_core", None)
    scheduler = getattr(core, "scheduler", None)
    kv_cache_config = getattr(scheduler, "kv_cache_config", None)
    groups = getattr(kv_cache_config, "kv_cache_groups", None) if kv_cache_config is not None else None
    group_sizes = []
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
        # so the group layout cannot be read in-process. Under mamba align the
        # patched attention block size equals the group lcm (the pool worker
        # startup log reports equal hash/lcm/block sizes), so fall back to the
        # cache block size; the H2 exact-hit assertion then still pins the real
        # behavior because a wrong fallback shows up as a token-count delta.
        group_sizes = [block_size]
    lcm = group_sizes[0]
    for size in group_sizes[1:]:
        lcm = lcm * size // math.gcd(lcm, size)
    return block_size, lcm


def _isolation_sentence() -> str:
    """Run-unique R4 isolation prompt base.

    The tier-3 storage tiers (UBS IO block device, Mooncake offload directory)
    PERSIST their data across service restarts -- measured on 80.48.37.145
    (2026-11-12): a gate-mode re-run's R4 disjoint prompt hit the *previous*
    run's same-content entries for a full 15,360-token prefix ("R4 isolation
    failed: ... got 15360.0"). That is legitimate persistent-pool reuse, not a
    key collision, but it defeats R4's fresh-pool assumption. Mixing a
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
    """Exact hit-token expectation for the GLM align cache layout.

    The generic ``expected_hit_token_count`` formula (floor(prompt / lcm) *
    lcm) is wrong for this engine: the scheduler **drops the final engine
    block before grouping the survivors into lcm chunks** (documented from
    measurement in PR #20757 case #11: "The scheduler drops the final engine
    block before grouping into lcm chunks ... 16,388 -> 16,128" at
    block_size=128 / lcm=1152, where the drop and the formula happened to
    agree). On the real W8A8 TP8 engine the align layout reports
    block_size == lcm == 640, and the drop now costs exactly one whole lcm
    group: 16,388 tokens carry 25 full blocks, the survivor set is 24 blocks
    = 15,360 tokens = 24 lcm chunks, measured on 80.48.37.145 (2026-11-12,
    glm53flash_tp8_memcache_bulk_tier2 record run: "hit tokens 15360 !=
    expected 16000" with a clean data plane). The pool served every block it
    was given; only this expectation needed the drop rule.
    """
    full_blocks = prompt_tokens // block_size
    if full_blocks <= 1:
        return 0
    survivor_tokens = (full_blocks - 1) * block_size
    return survivor_tokens // lcm * lcm


def _phase_hit(runner: VllmRunner, state: dict[str, Any], tag: str, reset_local: bool) -> None:
    """Phase 2: pool hit quality - R2/R3 exact hits, token identity, R4 isolation.

    ``reset_local`` is set only for the fallback leg-A mode (local prefix
    cache kept on because hybrid align rejects the strict combination): every
    round then drops just the local KV with ``reset_connector=False``, so the
    asserted hits can still only come from the pool and the pool content is
    never cleared between rounds.
    """
    prompt_a, prompt_b, cold = state["prompt_a"], state["prompt_b"], state["cold"]
    block_size, lcm = state["shapes"]
    expected = _expected_hit_tokens(len(prompt_a), block_size, lcm)
    assert expected > 0, (
        f"[{tag}] prompt {len(prompt_a)} cannot hit with block_size={block_size} lcm={lcm}; "
        "raise POOL_PROMPT_TOKENS above 2x the reported lcm"
    )

    def _round(prompt_ids: list[int], max_tokens: int = POOL_MAX_TOKENS) -> list[int]:
        if reset_local:
            runner.model.reset_prefix_cache(reset_connector=False)
        return _generate(runner, prompt_ids, max_tokens)

    if state["ssd"]:
        # Tier-3 pressure: SSD_PRESSURE_PROMPTS additional 16K prefixes push
        # utilization across the derived evict_threshold_high, spilling the
        # oldest ~20% of the shared prefix (LRU victim) down to the SSD tier.
        # After the settle window the hit rounds read back the mixed
        # ~80% DRAM / ~20% SSD prefix - the production-realistic mixed-hit
        # pattern.
        for sentence in PRESSURE_SENTENCES[:SSD_PRESSURE_PROMPTS]:
            _round(_natural_tokens(runner, sentence, POOL_PROMPT_TOKENS), 1)
        time.sleep(SSD_SETTLE_SECONDS)

    if reset_local:
        runner.model.reset_prefix_cache(reset_connector=False)
    began = time.perf_counter()
    before = snapshot_metrics(runner)
    hit1 = _round(prompt_a)
    mid = snapshot_metrics(runner)
    r2_seconds = time.perf_counter() - began
    assert_hit_tokens(before, mid, expected=expected, tag=f"{tag} R2")
    assert_output_identical(cold, hit1, tag=f"{tag} R2")

    began = time.perf_counter()
    hit2 = _round(prompt_a)
    after = snapshot_metrics(runner)
    r3_seconds = time.perf_counter() - began
    assert_hit_tokens(mid, after, expected=expected, tag=f"{tag} R3")
    assert_output_identical(cold, hit2, tag=f"{tag} R3")

    _round(prompt_b)
    final = snapshot_metrics(runner)
    assert_no_hit(after, final, tag=f"{tag} R4")

    state["hit"] = hit2
    state["hits_delta"] = after.hits - mid.hits
    state["expected_hits"] = expected
    print(
        f"[kv_pool_a2] {tag} hit: expected_tokens={expected} "
        f"r2_delta={mid.hits - before.hits:.0f} r3_delta={after.hits - mid.hits:.0f} "
        f"r2_latency={r2_seconds:.1f}s r3_latency={r3_seconds:.1f}s "
        f"queries_delta={(final.queries - before.queries):.0f}",
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
    state["hits_delta"] = 0.0  # the control leg owns local hits, not pool hits
    state["expected_hits"] = 0
    print(
        f"[kv_pool_a2] {tag} control: prefix_cache_hits_delta={after - before:.0f}",
        flush=True,
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _phase_perf(runner: VllmRunner, prompt_ids: list[int], tag: str, reset_local: bool) -> tuple[float, float, float]:
    """Phase 3: end-to-end latency of the repeatedly-hit prefix (TTFT proxy).

    ``reset_local`` (fallback leg A only) drops the local KV before every
    sample, otherwise every sample after the first would measure a local cache
    hit instead of a pool hit. The control leg intentionally keeps its warm
    local cache.
    """
    ttft: list[float] = []
    produced = 0
    started = time.perf_counter()
    for index in range(AB_WARMUP + AB_SAMPLES):
        if reset_local:
            runner.model.reset_prefix_cache(reset_connector=False)
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


def _emit_ab_record(record: dict[str, Any]) -> None:
    """Append the case record as one JSON line to the shared AB artifact."""
    with open(AB_ARTIFACT, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _leg_stats(p50: float, p95: float, throughput: float, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "acc": True,
        "hit_tokens": float(state["hits_delta"]),
        "expected_hits": state["expected_hits"],
        "ttft_p50": p50,
        "ttft_p95": p95,
        "throughput": throughput,
    }


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="kvpp,chunked_prefill,prefix_caching,mtp,spec_decode,sfa_dsa",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@pytest.mark.parametrize(
    ("use_layerwise",),
    [(False,), (True,)],
    ids=["memcache_bulk_tier2", "memcache_layerwise_tier3"],
)
@patch.dict(
    os.environ,
    {
        # V1 model runner: GLM5.3-Flash.md:25-26 (only V1 is supported for
        # this model; the Qwen3.8 A2 precedent runs V2 - intentional,
        # model-specific difference, see module docstring).
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # Fixed interpreter hash seed: memcache keys are content hashes and
        # every worker must compute them identically (GLM5.3-Flash.md:1449/
        # 1660 pool sections). Spawned worker processes read this on exec.
        "PYTHONHASHSEED": "0",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(rank) for rank in range(TP_SIZE)),
        # Pool-scale transfer/startup timeouts from the GLM5.3-Flash A2 pool
        # section (docs:1659-1661).
        "ASCEND_TRANSFER_TIMEOUT": "10000",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
    },
)
@wait_until_npu_memory_free()
def test_glm53_flash_kv_pool_ab_a2(use_layerwise: bool):
    tier_ssd = use_layerwise
    mode = "layerwise" if use_layerwise else "bulk"
    tier = "tier3-ssd" if tier_ssd else "tier2-dram"
    case = f"glm53flash_tp8_memcache_{mode}_{tier}"

    _require_a2_devices()
    _require_checkpoint()
    _require_memcache()
    if tier_ssd:
        _require_ubsio_disk()
    assert POOL_MAX_MODEL_LEN >= POOL_MIN_MAX_MODEL_LEN, (
        f"GLM53_POOL_MAX_MODEL_LEN={POOL_MAX_MODEL_LEN} cannot hold prompt {POOL_PROMPT_TOKENS} "
        f"+ {POOL_MAX_TOKENS} output tokens"
    )
    if tier_ssd:
        print(
            f"[kv_pool_a2] {case} tier3 sizing: dram_tier={SSD_DRAM_TIER}/rank "
            f"evict_high={MEMCACHE_EVICT_THRESHOLD_HIGH} evict_low={MEMCACHE_EVICT_THRESHOLD_LOW} "
            f"prompt_mb_planned={PROMPT_POOL_MB_PER_RANK} "
            f"pressure={SSD_PRESSURE_PROMPTS} ubsio_disk={MEMCACHE_UBSIO_DISK} ubsio_devices={MEMCACHE_UBSIO_DEVICES} "
            f"ubsio_mem_gb={MEMCACHE_UBSIO_MEM_GB}",
            flush=True,
        )

    def _run_pool_leg(local_cache: bool) -> dict[str, Any]:
        with _leg(True, use_layerwise, tier_ssd, local_cache, f"{case}-A") as runner:
            state: dict[str, Any] = {"ssd": tier_ssd}
            state["shapes"] = _engine_pool_shapes(runner)
            block_size, lcm = state["shapes"]
            print(
                f"[kv_pool_a2] {case} A: block_size={block_size} lcm={lcm} local_cache={local_cache} "
                f"(dummy reference lcm={DUMMY_REF_LCM} -> hits={DUMMY_REF_HITS}, never asserted)",
                flush=True,
            )
            _phase_accuracy(runner, state, f"{case}-A")
            _phase_hit(runner, state, f"{case}-A", reset_local=local_cache)
            state["perf"] = _phase_perf(runner, state["prompt_a"], f"{case}-A", reset_local=local_cache)
            return state

    # Leg A: strict first (pool on + local prefix cache off, the user-level
    # A/B). A startup rejection (hybrid align coupled to the local cache,
    # GLM5.3-Flash.md:1464-1470) downgrades BOTH legs to align + local cache
    # with per-round connector-preserving resets - never silently: the mode,
    # the engine error and the adjusted methodology land in the artifact and
    # the deviation log line.
    ab_mode = MODE_STRICT
    deviation_reason = ""
    try:
        a_state = _run_pool_leg(local_cache=False)
    except LegStartupError as exc:
        deviation_reason = str(exc)
        print(
            f"[kv_pool_a2] deviation case={case}: engine rejected leg A (pool on + "
            f"enable_prefix_caching=False with mamba_cache_mode=align); falling back to "
            f"both-legs-align + per-round reset_prefix_cache(reset_connector=False) on "
            f"leg A so the pool stays the only hit source. engine said: {deviation_reason}",
            flush=True,
        )
        time.sleep(AB_LEG_GAP_SECONDS)
        ab_mode = MODE_ALIGN_FALLBACK
        a_state = _run_pool_leg(local_cache=True)

    # Leave time for the pool service and worker processes to release NPU
    # memory before the control leg starts.
    time.sleep(AB_LEG_GAP_SECONDS)

    with _leg(False, use_layerwise, tier_ssd, True, f"{case}-B") as runner:
        b_state: dict[str, Any] = {"ssd": False, "shapes": a_state["shapes"]}
        _phase_accuracy(runner, b_state, f"{case}-B")
        _phase_control(runner, b_state, f"{case}-B")
        b_state["perf"] = _phase_perf(runner, b_state["prompt_a"], f"{case}-B", reset_local=False)

    a_p50, a_p95, a_throughput = a_state["perf"]
    b_p50, b_p95, b_throughput = b_state["perf"]
    token_identical = a_state["hit"] == b_state["hit"]
    tier_key = "tier3" if tier_ssd else "tier2"
    ttft_limit = TIER_TTFT_LIMITS[tier_key]
    throughput_limit = TIER_THROUGHPUT_LIMITS[tier_key]
    perf_ok = (
        a_p50 <= b_p50 * ttft_limit and a_p95 <= b_p95 * ttft_limit and a_throughput >= b_throughput * throughput_limit
    )
    verdict = "PASS" if (token_identical and perf_ok) else "FAIL"
    label = "PASS" if verdict == "PASS" else "POOL_DEFECT"
    delta_ttft = (a_p50 / b_p50 - 1.0) if b_p50 else 0.0
    delta_throughput = (a_throughput / b_throughput - 1.0) if b_throughput else 0.0

    record = {
        "case": case,
        "backend": "memcache",
        "mode": mode,
        "tier": tier,
        "ab_local_cache": ab_mode,
        "deviation_reason": deviation_reason or None,
        "block_size": a_state["shapes"][0],
        "lcm": a_state["shapes"][1],
        "expected_hits": a_state["expected_hits"],
        "tier_gates": {"ttft_limit": ttft_limit, "throughput_limit": throughput_limit},
        "a": _leg_stats(a_p50, a_p95, a_throughput, a_state),
        "b": _leg_stats(b_p50, b_p95, b_throughput, b_state),
        "delta_ttft": delta_ttft,
        "delta_throughput": delta_throughput,
        "token_identical": token_identical,
        "verdict": verdict,
        "label": label,
        "gate_mode": AB_MODE,
        "env": {
            "model": MODEL,
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER", ""),
            "kv_pool_ab_mode": AB_MODE,
            "gpu_memory_utilization": POOL_GPU_MEM_UTIL,
            "max_model_len": POOL_MAX_MODEL_LEN,
            "num_speculative_tokens": POOL_NUM_SPEC_TOKENS,
        },
    }
    _emit_ab_record(record)
    for leg, p50, p95, throughput, state in (
        ("A", a_p50, a_p95, a_throughput, a_state),
        ("B", b_p50, b_p95, b_throughput, b_state),
    ):
        print(
            f"[kv_pool_ab] case={case} leg={leg} acc=1 hit_tokens={state['hits_delta']:.0f} "
            f"ttft_p50={p50:.2f} ttft_p95={p95:.2f} throughput={throughput:.1f} "
            f"delta_ttft={delta_ttft:+.4f} delta_throughput={delta_throughput:+.4f} "
            f"verdict={verdict} label={label}",
            flush=True,
        )
    print(
        f"[kv_pool_a2] case={case} A/B: mode={ab_mode} ttft_p50 {a_p50:.1f} vs {b_p50:.1f} "
        f"({a_p50 / b_p50:.2f}x) ttft_p95 {a_p95:.1f} vs {b_p95:.1f} ({a_p95 / b_p95:.2f}x) "
        f"throughput {a_throughput:.0f} vs {b_throughput:.0f} "
        f"({a_throughput / b_throughput:.2f}x) token_identical={token_identical} "
        f"verdict={verdict} mode={AB_MODE} limits=ttft<={ttft_limit} tput>={throughput_limit}",
        flush=True,
    )
    assert token_identical, (
        f"[{case}] pooled leg and control leg outputs differ: pool hit path corrupted "
        f"the reused prefix (A tail={a_state['hit'][-8:]} B tail={b_state['hit'][-8:]})"
    )
    if AB_MODE == "gate":
        assert perf_ok, (
            f"[{case}] pooled leg performance outside A/B limits: "
            f"ttft_p50 {a_p50:.1f}ms vs {b_p50:.1f}ms, ttft_p95 {a_p95:.1f}ms vs {b_p95:.1f}ms, "
            f"throughput {a_throughput:.0f} vs {b_throughput:.0f} tok/s "
            f"(limits: ttft <= {ttft_limit}x, throughput >= {throughput_limit}x; mode={ab_mode})"
        )
