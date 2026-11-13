# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A2 KV-pool protection cases for the hybrid-GDN Qwen3.8-27B W8A8 model.

Four single-node ``kv_both`` scenarios guard the pooling matrix on A2:

======================  =========  =========  ===========
scenario                backend    mode       tier
======================  =========  =========  ===========
mooncake_bulk_tier2     mooncake   bulk       DRAM
mooncake_layerwise_tier3 mooncake  layerwise  DRAM + SSD
memcache_bulk_tier2     memcache   bulk       DRAM
memcache_layerwise_tier3 memcache layerwise  DRAM + SSD
======================  =========  =========  ===========

Every scenario runs two control legs, each inside one engine instance:

- Leg A: pool enabled (backend / transfer mode / storage tier per scenario),
  local prefix cache disabled;
- Leg B: pool disabled, local prefix cache enabled.

Both legs run three phases inside the single service: accuracy (cold greedy
output of a shared 16K natural-language prefix), pool hit (R2/R3 exact
hit-token count, token identity against the cold output and an R4
disjoint-prompt miss), and performance (TTFT-proxy sampling on the
repeatedly-hit prefix). The A/B comparison asserts that the pooled leg
reproduces the control leg's tokens exactly and stays within the per-tier
performance gates calibrated on A2 (see ``TIER_TTFT_LIMITS`` /
``TIER_THROUGHPUT_LIMITS``).

The tier-3 scenarios arm the SSD storage tier but keep it eviction-free
(deliberately, see the tier sizing comment below): the pool DRAM tier holds the
whole working set so hits are served from DRAM, and the SSD tier is armed and
receives the writes but its read-back path is not exercised. The memcache
tier-3 legs use UBS IO storage (raw block device, rewarm readbacks from the
UBS IO DRAM buffer) and the mooncake tier-3 leg arms SSD offload + the SSD ->
DRAM prefetch per the official KV pool guide (activated only on mooncake
builds exposing ``ExistOptions``; on older builds the DRAM-only tier runs and
the SSD store stays armed-but-idle).

Environment knobs:

- ``QWEN38_POOL_MODEL``: model id or local checkpoint path (default
  ``Eco-Tech/Qwen3.8-27B-w8a8``);
- ``KV_POOL_AB_MODE``: ``gate`` (default, A/B comparison must pass) or
  ``record`` (only log the verdict);
- ``KV_POOL_MEMCACHE_UBSIO_DISK``: raw block device passed to the container
  for the memcache tier-3 UBS IO storage (default ``/dev/ubsio0``; loop
  devices are a documented option — dedicated, unmounted, no filesystem
  signature);
- ``KV_POOL_MEMCACHE_UBSIO_DEVICES``: UBS IO standalone device count (default
  ``TP_SIZE``, one per DRAM-enabled local service).

Host / runner prerequisites (official KV pool guide): the mooncake legs run
the HIXL RoCE direct mode (``HCCL_INTRA_ROCE_ENABLE=1``, set by the test)
over reserved 2MB hugepages (the guide reserves 200000 on the host; the
test skips when none are visible to the container), and the memcache tier-3
legs need the raw UBS IO block device above (the test skips with provisioning
steps when it is missing).
"""

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
from vllm import SamplingParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig, MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import (
    SingleNodeMemcacheManager,
    SingleNodeMooncakeManager,
)
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    AB_SAMPLES,
    AB_WARMUP,
    PREFIX_HITS_METRIC,
    assert_hit_tokens,
    assert_no_hit,
    assert_output_identical,
    expected_hit_token_count,
    snapshot_metrics,
)

MODEL = os.environ.get("QWEN38_POOL_MODEL", "Eco-Tech/Qwen3.8-27B-w8a8")
TP_SIZE = 2

# 16K-class shared prefix with a non-zero tail: the pool only transfers whole
# blocks/lcm groups, and an lcm-aligned prompt would trim its final block into
# a guaranteed miss (see kv_pool_watch.expected_hit_token_count).
POOL_PROMPT_TOKENS = 16388
# Fix the chunked-prefill shape for cold and hit rounds alike; 16,388 tokens
# prefill in [16384, 4] chunks in every leg.
POOL_MAX_NUM_BATCHED_TOKENS = 16384
POOL_MAX_MODEL_LEN = 16640
POOL_MAX_TOKENS = 16

# Tier sizing: a 16K prompt holds roughly 600MB of KV per rank, so the
# 4GB-per-rank pool DRAM tiers keep the whole working set resident and
# eviction-free in every scenario, and the tier-3 legs stay eviction-free on
# purpose. Verified on A2 (memcache_hybrid 1.3.0 + device_rdma + UBS IO),
# every path that moves data out of DRAM onto the SSD medium breaks the
# read-back, so the eviction-free design is a platform constraint, not a
# choice: (1) L2 eviction (evict_threshold watermarks) loses evicted blocks'
# GVA entries while the query still reports them as hits ("load_gvas: got
# invalid gva=0 ... load failed" -> HTTP 500); (2) an undersized UBS IO DRAM
# buffer (ubsio.mem=1GB/die) collapses the query index to 1-3 of 16 blocks
# (device reads stay 0); (3) non-layerwise + storage.enabled=true collapses
# cross-request hits to a single block regardless of buffer size or wait.
# The eviction-free tier-3 legs keep the SSD tier armed (writes reach the
# UBS IO layer; reads return through its DRAM buffer via rewarm) with the
# SSD medium read-back left to the platform fix and a Nightly layer.
DRAM_TIER = "4GB"
MEMCACHE_SSD_DRAM_TIER = "4GB"
MOONCAKE_SSD_DRAM_TIER = "4GB"

AB_MODE = os.environ.get("KV_POOL_AB_MODE", "gate")
# The mooncake-layerwise scenario runs without a version gate: the official
# support matrix asks for mooncake >= 0.3.14, but the layerwise DRAM path was
# validated empirically on the image's 0.3.13 and the gate was opened at the
# maintainer's direction. The version is printed per case for diagnostics,
# and the SSD offload + prefetch below only activates on builds exposing
# ExistOptions (mooncake >= 0.3.14, Mooncake PR 4272).
# Performance gates are per storage tier and calibrated on A2 (Qwen3.8-27B
# W8A8, TP2): against the local prefix-cache control leg the pooled leg
# transfers ~1GB per request through the pool, which costs 5-25% TTFT and
# 4-17% throughput (observed p50/p95 ratios across repeated runs on the
# shared machine: 1.05x-1.23x TTFT, 0.83x-0.96x throughput). The gates keep
# headroom for that inherent cost plus shared-machine noise while still
# failing catastrophic regressions (a full re-prefill miss runs ~12x slower
# and ~0.1x throughput). The tier-3 layerwise path is intrinsically slower
# (streamed per-layer loads, observed 0.42x-0.46x throughput for memcache)
# so it only gates catastrophic regressions. Both limits stay overridable
# via TIER_TTFT_LIMITS / TIER_THROUGHPUT_LIMITS edits for recalibration.
TIER_TTFT_LIMITS = {"tier2": 1.35, "tier3": 3.0}
TIER_THROUGHPUT_LIMITS = {"tier2": 0.75, "tier3": 0.30}
MEMCACHE_UBSIO_DISK = os.environ.get("KV_POOL_MEMCACHE_UBSIO_DISK", "/dev/ubsio0")
MEMCACHE_UBSIO_DEVICES = int(os.environ.get("KV_POOL_MEMCACHE_UBSIO_DEVICES", TP_SIZE))

# Natural-language prompts: random token ids flip under batch-shape
# differences even with intact KV, so H3/accuracy must use low-entropy text.
SENTENCE_A = "Explain how historical records help people understand the past. "
SENTENCE_B = "The mountain railway climbs steadily toward the old observatory. "


def _mooncake_version() -> str:
    try:
        return _pkg_version("mooncake-transfer-engine-npu")
    except PackageNotFoundError:
        return "0"


def _hugepages_total() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("HugePages_Total:"):
                    return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _mooncake_supports_ssd_prefetch() -> bool:
    """True when the installed Mooncake exposes ExistOptions (>= 0.3.14 builds).

    The official KV pool guide activates the SSD -> DRAM prefetch only on
    Mooncake builds exposing ``ExistOptions`` (Mooncake PR 4272); on builds
    without it the prefetch keys are ignored and the SSD-offloaded layerwise
    read path is unstable, so the tier-3 leg falls back to the DRAM-only
    tier there.
    """
    try:
        from mooncake.store import ExistOptions  # type: ignore[import-not-found]  # noqa: F401

        return hasattr(ExistOptions(), "prefetch_to_memory")
    except (ImportError, TypeError, AttributeError):
        # A missing module, a constructor that requires arguments in this
        # Mooncake version, or a renamed option set all mean the build does
        # not expose the SSD prefetch.
        return False


def _mooncake_pool_config(tier_ssd: bool, case_name: str) -> MooncakeKVPoolConfig:
    config: dict[str, Any] = {
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "ascend",
        "device_name": "",
        "global_segment_size": MOONCAKE_SSD_DRAM_TIER if tier_ssd else DRAM_TIER,
        "local_buffer_size": "1GB",
    }
    master_extra_args: tuple[str, ...] = ()
    if tier_ssd and _mooncake_supports_ssd_prefetch():
        # Tier 3 per the official KV pool guide: SSD offload plus the
        # independent SSD -> DRAM prefetch on exists checks. The prefetch
        # promotes SSD-only objects to DRAM during cache probes
        # (ExistOptions(prefetch_to_memory=True), Mooncake PR 4272), which is
        # what makes the SSD read path usable. The master must advertise
        # offload availability (--enable_offload=true) or every worker's
        # disk-segment mount fails with UNABLE_OFFLOAD and the store init
        # aborts. The offload directory must exist and be writable before
        # startup; the connector creates per-rank subdirectories under it.
        # Setting any prefetch parameter also makes the store create its own
        # TransferEngine instead of sharing the global one. On builds without
        # ExistOptions (mooncake 0.3.13) this stays off: the store write-
        # throughs every block straight to SSD ("Mem Storage: 0 B / 8.00 GB |
        # SSD Storage: 6.04 GB") and the intra-request layer reload of the
        # chunked prefill then fails intermittently ("Hybrid layerwise load
        # failed"), so the DRAM-only tier runs there instead (the SSD store
        # stays armed-but-idle).
        ssd_root = os.path.join(tempfile.gettempdir(), f"{case_name}_mooncake_ssd")
        os.makedirs(ssd_root, exist_ok=True)
        config.update(
            {
                "enable_ssd_offload": True,
                "ssd_offload_path": ssd_root,
                "enable_ssd_prefetch": True,
                "ssd_prefetch_cooldown_sec": 5,
                "ssd_prefetch_dedup_ttl_sec": 30,
            }
        )
        master_extra_args = ("--enable_offload=true",)
    return MooncakeKVPoolConfig(
        master_port=get_open_port(),
        metrics_port=get_open_port(),
        config=config,
        master_extra_args=master_extra_args,
    )


def _memcache_pool_config(tier_ssd: bool) -> MemcacheKVPoolConfig:
    local: dict[str, Any] = {
        "ock.mmc.log_level": "error",
        "ock.mmc.local_service.world_size": 256,
        # Official A2 recommendation (KV pool guide): device_rdma (RoCE).
        "ock.mmc.local_service.protocol": "device_rdma",
        "ock.mmc.local_service.dram.size": DRAM_TIER,
    }
    meta: dict[str, Any] = {
        "ock.mmc.log_level": "error",
        # The meta service also starts an HTTP metrics endpoint. Pin it to a
        # free port: the default 8000 collides with host services under
        # --network=host.
        "ock.mmc.meta_service.metrics_url": f"http://127.0.0.1:{get_open_port()}",
    }
    if tier_ssd:
        # Tier 3: UBS IO manages a raw (unmounted) block device as the SSD
        # tier. The 8GB/die UBS IO DRAM buffer holds the whole working set, so
        # reads return through it via rewarm while the SSD medium stays
        # armed-but-idle (deliberate; see the tier sizing comment above — the
        # eviction/overflow paths to the medium are broken on this stack and
        # the DRAM tier is sized to never trigger them). No evict watermarks:
        # the defaults never trigger at this utilization and explicit ones
        # only activate the broken eviction path.
        local["ock.mmc.local_service.storage.enabled"] = True
        local["ubsio.disk.path"] = MEMCACHE_UBSIO_DISK
        local["ubsio.mem.size_in_gb"] = 8
        local["ubsio.standalone.device_count"] = MEMCACHE_UBSIO_DEVICES
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={"meta": meta, "local": local},
    )


@contextmanager
def _leg(
    use_pool: bool,
    backend: str,
    use_layerwise: bool,
    tier_ssd: bool,
    case_name: str,
):
    """Build one leg: A turns the pool on with the local cache off, B the opposite.

    Everything else (graph mode per scenario, parallel layout, scheduler
    budget) stays identical across the legs so the only variable is where the
    reused prefix comes from: the pool or the local prefix cache.
    """
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            dtype="auto",
            quantization="ascend",
            tensor_parallel_size=TP_SIZE,
            distributed_executor_backend="mp",
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=POOL_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=0.85,
            seed=1024,
            enable_prefix_caching=not use_pool,
            enable_chunked_prefill=True,
            disable_log_stats=False,
            async_scheduling=True,
            mamba_cache_mode="align",
            compilation_config={"cudagraph_mode": "PIECEWISE" if use_layerwise else "FULL_DECODE_ONLY"},
        )
        if use_pool:
            if backend == "mooncake":
                # Official A2 pool environment (KV pool guide): A2 (and A3 on
                # RoCE links) runs the HIXL RoCE direct mode selected by
                # HCCL_INTRA_ROCE_ENABLE=1. The guide's multi-node script also
                # binds HCCL_IF_IP / *_SOCKET_IFNAME to a reachable NIC and
                # reserves 200000 2MB hugepages (a host/runner prerequisite;
                # the single-node legs keep the loopback route and only need
                # the pages to exist). Without RoCE direct mode the default
                # HCCS P2P channel fails cross-rank with DisableP2P=17 / IPC
                # import 107000 (put error -800). ASCEND_ENABLE_USE_FABRIC_MEM
                # must stay unset: fabric mem is the documented A3-HCCS mode,
                # and an empty-string value still reads as "set" to the
                # engine, which then fails its hugepage-reserve path on A2.
                os.environ.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
                stack.enter_context(patch.dict(os.environ, {"HCCL_INTRA_ROCE_ENABLE": "1"}))
                mooncake_config = _mooncake_pool_config(tier_ssd, case_name)
                ssd_root = mooncake_config.config.get("ssd_offload_path")
                if ssd_root:
                    # The tier-3 offload directory can hold gigabytes of KV
                    # blocks; remove it when the leg exits so repeated CI runs
                    # do not leak disk space.
                    stack.callback(shutil.rmtree, ssd_root, ignore_errors=True)
                pool = stack.enter_context(SingleNodeMooncakeManager(mooncake_config, case_name))
            else:
                pool = stack.enter_context(SingleNodeMemcacheManager(_memcache_pool_config(tier_ssd), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            extra_config: dict[str, Any] = {
                "backend": backend,
                "use_layerwise": use_layerwise,
                "lookup_rpc_port": "0",
            }
            if not use_layerwise:
                # Hybrid GDN needs the synchronous mamba state
                # copy-after-load on the bulk path; async loads skip it and
                # silently corrupt the restored state.
                extra_config["load_async"] = False
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


def _engine_pool_shapes(runner: VllmRunner) -> tuple[int, int]:
    """Return ``(block_size, lcm of KV-group block sizes)`` from the engine.

    Hybrid QSA/GDN models keep several KV cache groups with different block
    sizes; the pool only stores whole lcm groups, so the exact hit-token
    expectation must come from the engine's real group layout. The scheduler
    holds the ``KVCacheConfig`` produced by the engine core's cache
    initialization (see vllm EngineCore -> Scheduler(kv_cache_config=...)).
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
        # so the group layout cannot be read in-process. The hybrid align
        # layout keeps every KV cache group at the patched attention block
        # size (the pool worker startup log reports equal hash/lcm/block
        # sizes, e.g. "num_kv_cache_groups: 4, hash_block_size: 896,
        # lcm_block_size: 896"), so fall back to the cache block size.
        group_sizes = [block_size]
    lcm = group_sizes[0]
    for size in group_sizes[1:]:
        lcm = lcm * size // math.gcd(lcm, size)
    return block_size, lcm


def _phase_accuracy(runner: VllmRunner, state: dict[str, Any], tag: str) -> None:
    """Phase 1: cold greedy output of the shared prefix (also the pool's R1 write)."""
    started = time.perf_counter()
    state["prompt_a"] = _natural_tokens(runner, SENTENCE_A, POOL_PROMPT_TOKENS)
    state["prompt_b"] = _natural_tokens(runner, SENTENCE_B, POOL_PROMPT_TOKENS)
    state["cold"] = _generate(runner, state["prompt_a"], POOL_MAX_TOKENS)
    print(
        f"[kv_pool_a2] {tag} accuracy: cold_tokens={len(state['cold'])} "
        f"head={state['cold'][:6]} tail={state['cold'][-6:]} "
        f"cold_latency={(time.perf_counter() - started):.1f}s",
        flush=True,
    )


def _phase_hit(runner: VllmRunner, state: dict[str, Any], tag: str) -> None:
    """Phase 2: pool hit quality — R2/R3 exact hits, token identity, R4 isolation."""
    prompt_a, prompt_b, cold = state["prompt_a"], state["prompt_b"], state["cold"]
    block_size, lcm = state["shapes"]
    expected = expected_hit_token_count(len(prompt_a), block_size, lcm)
    assert expected > 0, (
        f"[{tag}] prompt {len(prompt_a)} cannot hit with block_size={block_size} lcm={lcm}; "
        "raise POOL_PROMPT_TOKENS above 2x the reported lcm"
    )

    began = time.perf_counter()
    before = snapshot_metrics(runner)
    hit1 = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    mid = snapshot_metrics(runner)
    r2_seconds = time.perf_counter() - began
    assert_hit_tokens(before, mid, expected=expected, tag=f"{tag} R2")
    assert_output_identical(cold, hit1, tag=f"{tag} R2")

    began = time.perf_counter()
    hit2 = _generate(runner, prompt_a, POOL_MAX_TOKENS)
    after = snapshot_metrics(runner)
    r3_seconds = time.perf_counter() - began
    assert_hit_tokens(mid, after, expected=expected, tag=f"{tag} R3")
    assert_output_identical(cold, hit2, tag=f"{tag} R3")

    _generate(runner, prompt_b, POOL_MAX_TOKENS)
    final = snapshot_metrics(runner)
    assert_no_hit(after, final, tag=f"{tag} R4")

    state["hit"] = hit2
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


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="kvpp,chunked_prefill,prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="full_and_piecewise",
)
@pytest.mark.parametrize(
    ("backend", "use_layerwise"),
    [
        ("mooncake", False),
        ("mooncake", True),
        ("memcache", False),
        ("memcache", True),
    ],
    ids=["mooncake_bulk_tier2", "mooncake_layerwise_tier3", "memcache_bulk_tier2", "memcache_layerwise_tier3"],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free()
def test_qwen38_kv_pool_ab_a2(backend: str, use_layerwise: bool):
    tier_ssd = use_layerwise
    mode = "layerwise" if use_layerwise else "bulk"
    tier = "tier3-ssd" if tier_ssd else "tier2-dram"
    case = f"qwen38_tp2_{backend}_{mode}_{tier}"

    if backend == "mooncake":
        # The official support matrix asks for mooncake >= 0.3.14 for
        # layerwise; the layerwise path (DRAM and the SSD offload + prefetch
        # readback) was validated empirically on the image's 0.3.13 and the
        # gate was opened at the maintainer's direction. Print the
        # version for diagnostics.
        print(f"[kv_pool_a2] {case}: mooncake {_mooncake_version()}", flush=True)
        if _hugepages_total() == 0:
            pytest.skip(
                "mooncake pooling on A2 requires reserved 2MB hugepages (the official "
                "guide reserves 200000); none are visible to the container. Reserve them "
                "on the host/runner before running this test."
            )
    if backend == "memcache" and tier_ssd and not os.path.exists(MEMCACHE_UBSIO_DISK):
        pytest.skip(
            f"memcache tier-3 needs the raw UBS IO block device {MEMCACHE_UBSIO_DISK!r} "
            "(no mount point, no filesystem signature; a loop device is fine — provision "
            "one on the host, e.g. truncate -s 64G file && losetup -f --show file, and "
            "pass it into the container with --device; set KV_POOL_MEMCACHE_UBSIO_DISK)."
        )

    with _leg(True, backend, use_layerwise, tier_ssd, case) as runner:
        a_state: dict[str, Any] = {}
        a_state["shapes"] = _engine_pool_shapes(runner)
        print(f"[kv_pool_a2] {case} A: block_size={a_state['shapes'][0]} lcm={a_state['shapes'][1]}", flush=True)
        _phase_accuracy(runner, a_state, f"{case}-A")
        _phase_hit(runner, a_state, f"{case}-A")
        a_p50, a_p95, a_throughput = _phase_perf(runner, a_state["prompt_a"], f"{case}-A")

    # Leave time for the pool service and worker processes to release NPU
    # memory before the control leg starts.
    time.sleep(10)

    with _leg(False, backend, use_layerwise, tier_ssd, case) as runner:
        b_state: dict[str, Any] = {}
        b_state["shapes"] = a_state["shapes"]
        _phase_accuracy(runner, b_state, f"{case}-B")
        _phase_control(runner, b_state, f"{case}-B")
        b_p50, b_p95, b_throughput = _phase_perf(runner, b_state["prompt_a"], f"{case}-B")

    token_identical = a_state["hit"] == b_state["hit"]
    ttft_limit = TIER_TTFT_LIMITS["tier3" if tier_ssd else "tier2"]
    throughput_limit = TIER_THROUGHPUT_LIMITS["tier3" if tier_ssd else "tier2"]
    perf_ok = (
        a_p50 <= b_p50 * ttft_limit and a_p95 <= b_p95 * ttft_limit and a_throughput >= b_throughput * throughput_limit
    )
    verdict = "PASS" if (token_identical and perf_ok) else "FAIL"
    print(
        f"[kv_pool_a2] case={case} A/B: ttft_p50 {a_p50:.1f} vs {b_p50:.1f} "
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
            f"(limits: ttft <= {ttft_limit}x, throughput >= {throughput_limit}x)"
        )
