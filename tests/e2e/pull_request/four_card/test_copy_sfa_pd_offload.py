# SPDX-License-Identifier: Apache-2.0
"""A3 smoke test: regular Prefill KV -> Decode host KV with fused Copy-SFA."""

import json
from pathlib import Path

import pytest
import requests
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggPDProxy, RemotePDServer, wait_until_npu_memory_free
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
OUTPUT_TOKENS = 16
TOPK = 2048
LAYERWISE_PROXY = (
    Path(__file__).resolve().parents[4]
    / "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
)

pytestmark = [
    pytest.mark.e2e_model(MODEL),
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Copy-SFA offload requires A3"),
]


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa,cpu_offloading",
    parallel="TP,EP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_regular_prefill_copy_sfa_decode():
    """Exercise host KV reads, sparse selection and a reused request slot."""
    model = maybe_model_redirect(MODEL)
    p_port, d_port, proxy_port = get_open_port(), get_open_port(), get_open_port()
    common_args = [
        "--model",
        model,
        "--trust-remote-code",
        "--enable-request-id-headers",
        "--quantization",
        "ascend",
        "--enable-expert-parallel",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--block-size",
        "128",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "4096",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--seed",
        "42",
        "--generation-config",
        "vllm",
    ]
    servers = []
    for port, role, kv_port in ((p_port, "kv_producer", 30400), (d_port, "kv_consumer", 30600)):
        servers.append(
            [
                *common_args,
                "--port",
                str(port),
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "SfaRemoteD2HConnector",
                        "kv_role": role,
                        "kv_port": kv_port,
                        "kv_connector_extra_config": {"transfer_backend": "memfabric", "use_layerwise": True},
                    }
                ),
            ]
        )
    # P owns ordinary device KV. No AscendStore, shared buffers or sparse offload.
    servers[0] += ["--enforce-eager"]
    servers[1] += [
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}),
        "--additional-config",
        json.dumps(
            {
                "sparse_kv_offload_config": {
                    "enabled": True,
                    "fused_op_type": "fused_copy_sfa",
                    "topk_buffer_size": TOPK,
                    "dram_size_per_dp_GB": 2,
                    "keep_device_kv_cache": False,
                    "use_fused_overlap": False,
                }
            }
        ),
    ]
    short_prompt = "The capital of France is"
    # Exceed index_topk so the test exercises sparse selection, not just the
    # dense short-sequence path. Check the actual tokenized length below.
    long_prompt = "This is background information. " * 500 + short_prompt
    with (
        RemotePDServer(servers, env_dict={"VLLM_USE_V2_MODEL_RUNNER": "0"}),
        DisaggPDProxy(
            port=proxy_port,
            prefiller_ports=[p_port],
            decoder_ports=[d_port],
            proxy_script=LAYERWISE_PROXY,
        ) as proxy,
    ):
        # A short request after a long one reuses the only request slot.
        for prompt in (short_prompt, long_prompt, short_prompt):
            response = requests.post(
                proxy.url_for("v1", "completions"),
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": 0,
                    "max_tokens": OUTPUT_TOKENS,
                    "ignore_eos": True,
                    "stream": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            assert "error" not in result, result
            assert len(result["choices"]) == 1, result
            choice = result["choices"][0]
            assert choice["finish_reason"] == "length", result
            assert choice.get("stop_reason") != "recomputed", result
            assert isinstance(choice["text"], str) and choice["text"], result
            # More than the prefill token must be produced by the decode path.
            assert result["usage"]["completion_tokens"] == OUTPUT_TOKENS, result
            if prompt == long_prompt:
                assert TOPK < result["usage"]["prompt_tokens"] <= 4096 - OUTPUT_TOKENS, result
