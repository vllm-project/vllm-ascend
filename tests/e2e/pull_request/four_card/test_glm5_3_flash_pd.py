# SPDX-License-Identifier: Apache-2.0
"""PCP2/EP2 producer and DP2/EP2 consumer, using nine synthetic Flash layers.

This is functional coverage, not checkpoint accuracy or MTP acceptance quality.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.glm53flash_pd_checks import check_completion, check_state_pulls
from tests.e2e.conftest import RemotePDServer, wait_until_npu_memory_free
from tests.e2e.pull_request.four_card.test_glm5_3_flash import _write_model

MODEL_NAME = "flash-pd-test"
OUTPUT_TOKENS = 16


class FlashPDServer(RemotePDServer):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            # A constructor failure never enters the context manager.
            self._terminate_server()
            raise

    def _start_server_with_prefix(self, server_cmd, env_dict, log_prefix):
        # Import the test loader before parsing the CLI and in spawned workers.
        command = [sys.executable, "-m", "tests.e2e.common.glm53flash_pd", *server_cmd[1:]]
        return super()._start_server_with_prefix(command, env_dict, log_prefix)


def serve_args(model, port, kv_port, producer, eager, prefix_caching):
    topology = {
        "prefill": {"dp_size": 1, "tp_size": 1, "pcp_size": 2},
        "decode": {"dp_size": 2, "tp_size": 1, "pcp_size": 1},
    }
    args = [
        str(model),
        "--served-model-name",
        MODEL_NAME,
        "--port",
        str(port),
        "--load-format",
        "glm53flash_pd_dummy",
        "--worker-extension-cls",
        "tests.e2e.common.glm53flash_pd.FlashPDWorker",
        "--dtype",
        "bfloat16",
        "--quantization",
        "ascend",
        "--kv-cache-dtype",
        "fp8",
        "--attention-config.indexer_kv_dtype",
        "fp8",
        "--tensor-parallel-size",
        "1",
        "--prefill-context-parallel-size",
        "2" if producer else "1",
        "--data-parallel-size",
        "1" if producer else "2",
        "--enable-expert-parallel",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "512",
        "--block-size",
        "128",
        "--num-gpu-blocks-override",
        "64",
        "--gpu-memory-utilization",
        "0.8",
        "--seed",
        "1024",
        "--generation-config",
        "vllm",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0, "video": 0}),
        "--enable-chunked-prefill",
        "--use-replayssm",
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        "--mamba-cache-mode",
        "align" if prefix_caching else "none",
        "--speculative-config",
        json.dumps({"method": "deepseek_mtp", "num_speculative_tokens": 3, "enforce_eager": producer or eager}),
        "--additional-config",
        json.dumps({"enable_cpu_binding": False, "enable_fused_mc2": 0}),
        "--kv-transfer-config",
        json.dumps(
            {
                "kv_connector": "MooncakeConnectorV2",
                "kv_role": "kv_producer" if producer else "kv_consumer",
                "kv_port": kv_port,
                "kv_connector_extra_config": topology,
            }
        ),
    ]
    if producer or eager:
        args.append("--enforce-eager")
    else:
        args += [
            "--compilation-config",
            json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16]}),
        ]
    return args


def complete(url, prompt, dp_rank=None, **kwargs):
    response = requests.post(
        url + "/v1/completions",
        headers={"X-data-parallel-rank": str(dp_rank)} if dp_rank is not None else {},
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            **kwargs,
        },
        timeout=180,
    )
    assert response.ok, f"HTTP {response.status_code}: {response.text}"
    result = response.json()
    assert "error" not in result, result
    return result


def rpc(url, method, *args):
    response = requests.post(url + "/collective_rpc", json={"method": method, "args": list(args)}, timeout=60)
    response.raise_for_status()
    return [
        item for result in response.json()["results"] for item in (result if isinstance(result, list) else [result])
    ]


@pytest.mark.parametrize("eager,prefix_caching", [(True, False), (False, True)], ids=["eager", "graph-prefix"])
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,chunked_prefill,prefix_caching,logprobs",
    parallel="PCP,DP,EP",
    deploy="pd_disaggregation",
    hardware="A5",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@wait_until_npu_memory_free()
def test_glm53flash_pcp2_dp2(tmp_path, eager, prefix_caching):
    model = tmp_path / "model"
    _write_model(model)
    p_port, d_port = get_open_port(), get_open_port()
    p_url, d_url = f"http://127.0.0.1:{p_port}", f"http://127.0.0.1:{d_port}"
    servers = [
        serve_args(model, p_port, 26770, True, eager, prefix_caching),
        serve_args(model, d_port, 26970, False, eager, prefix_caching),
    ]
    with FlashPDServer(
        servers,
        env_dict={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_BUFFSIZE": "400",
        },
        max_wait_seconds=900,
    ):
        for url, expected in ((p_url, (1, 2, 1, 2)), (d_url, (1, 1, 2, 2))):
            workers = rpc(url, "flash_pd_inspect")
            assert workers
            assert len(workers) == 2
            assert {worker["dp_rank"] for worker in workers} == set(range(expected[2]))
            for worker in workers:
                assert tuple(worker[key] for key in ("tp", "pcp", "dp", "ep")) == expected
                assert worker["kda_heads"] == [64 // expected[1]] * 7
                assert worker["replayssm"] and worker["has_speculator"]
                assert worker["mtp_tokens"] == 3
                assert worker["prefix_caching"] == prefix_caching
                assert worker["graph_mode"] == ("NONE" if url == p_url or eager else "FULL_DECODE_ONLY")
        rpc(d_url, "flash_pd_probe")

        def generate(length, dp_rank=0):
            prompt = [10 + i % 1000 for i in range(length)]
            prefill = complete(p_url, prompt, max_tokens=1, kv_transfer_params={"do_remote_decode": True})
            transfer = prefill.get("kv_transfer_params")
            assert transfer and transfer.get("do_remote_prefill"), prefill
            assert any(transfer.get("remote_block_ids", [])), transfer
            result = complete(d_url, prompt, dp_rank=dp_rank, kv_transfer_params=transfer)
            check_completion(result, OUTPUT_TOKENS)
            return result

        generate(128)
        check_state_pulls(rpc(d_url, "flash_pd_records"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(generate, length, index % 2) for index, length in enumerate([127, 129, 511, 512, 513])
            ]
            for future in futures:
                future.result()
        records = rpc(d_url, "flash_pd_records")
        assert all(worker["copies"] for worker in records), "Both decode DP ranks must receive remote state"
        check_state_pulls(records)
