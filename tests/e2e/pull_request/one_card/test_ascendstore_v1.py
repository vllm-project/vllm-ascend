# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise in-process AscendStore v1 against a real Mooncake store."""

import shutil
from pathlib import Path

import pytest
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_ports_list

from tests.e2e.common.kv_pool.ascendstore_v1_probe import (
    clear_worker_local_kv,
    collect_worker_io_probe,
    install_worker_io_probe,
)
from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.conftest import VllmRunner
from tests.e2e.nightly.single_node.models.scripts.kv_pool_runtime import SingleNodeMooncakeManager

MODEL = "Qwen/Qwen3-0.6B"
CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector"
PROMPT_TOKEN_COUNT = 513
OUTPUT_TOKEN_COUNT = 8
POOL_MEMORY_BYTES = 1 << 30

pytestmark = pytest.mark.e2e_model(MODEL)


def test_inprocess_store_lookup_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Store a cold prompt, discard local KV, then verify remote reuse."""
    assert shutil.which("mooncake_master") is not None, "The smoke test requires mooncake_master on PATH"
    master_port, metrics_port, lookup_port = get_open_ports_list(3)
    pool_config = MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "device_name": "",
            "global_segment_size": POOL_MEMORY_BYTES,
            "local_buffer_size": POOL_MEMORY_BYTES,
        },
        master_port=master_port,
        metrics_port=metrics_port,
    )
    transfer_config = KVTransferConfig(
        kv_connector="AscendStoreV1Connector",
        kv_connector_module_path=CONNECTOR_MODULE,
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={
            "backend": "mooncake",
            "lookup_rpc_port": lookup_port,
            "use_layerwise": False,
            "load_async": False,
            "save_decode_cache": False,
        },
    )
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    # Trusted test callbacks need pickle support in the client and spawned EngineCore.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Keep a nonaligned tail: Store publishes full chunks; Load still leaves work for prefill.
    prompts = [TokensPrompt(prompt_token_ids=[0] * PROMPT_TOKEN_COUNT)]
    sampling_params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKEN_COUNT, ignore_eos=True)

    with SingleNodeMooncakeManager(pool_config, tmp_path.name) as pool:
        for name, value in pool.server_envs.items():
            monkeypatch.setenv(name, value)
        with VllmRunner(
            maybe_model_redirect(MODEL),
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.5,
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            enforce_eager=True,
            async_scheduling=False,
            seed=42,
            kv_transfer_config=transfer_config,
        ) as runner:
            (granularity,) = runner.model.collective_rpc(install_worker_io_probe)
            expected_loaded_tokens = PROMPT_TOKEN_COUNT // granularity * granularity
            assert 0 < expected_loaded_tokens < PROMPT_TOKEN_COUNT
            cold_output = runner.model.generate(prompts, sampling_params, use_tqdm=False)[0]
            assert cold_output.finished
            assert cold_output.num_cached_tokens == 0
            assert len(cold_output.outputs[0].token_ids) == OUTPUT_TOKEN_COUNT

            # Reset removes cache hashes, not bytes. Erase Worker KV as well, but preserve the Mooncake objects.
            assert runner.model.reset_prefix_cache(), "Local KV must be cleared before testing remote Load"
            (cold_evidence,) = runner.model.collective_rpc(clear_worker_local_kv)
            assert len(cold_evidence["stored_keys"]) == expected_loaded_tokens // granularity
            warm_output = runner.model.generate(prompts, sampling_params, use_tqdm=False)[0]
            assert warm_output.finished
            assert warm_output.num_cached_tokens == expected_loaded_tokens
            assert warm_output.outputs[0].token_ids == cold_output.outputs[0].token_ids
            (warm_evidence,) = runner.model.collective_rpc(collect_worker_io_probe)
            assert warm_evidence["get_calls"] == 1, "The warm request must execute a real Backend GET"
            assert sorted(warm_evidence["loaded_keys"]) == sorted(cold_evidence["stored_keys"])
            assert warm_evidence["loaded_ranges"] == [
                [start, start + granularity] for start in range(0, expected_loaded_tokens, granularity)
            ]
            assert warm_evidence["loaded_bytes"] == cold_evidence["stored_bytes"] > 0
            print(
                f"AscendStore v1 verified: GET calls={warm_evidence['get_calls']}, "
                f"keys={len(warm_evidence['loaded_keys'])}, tokens={expected_loaded_tokens}, "
                f"bytes={warm_evidence['loaded_bytes']}, local KV erased, exact source bytes restored"
            )
