# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real reduced DSV4 Hybrid PD: retain both compressor ratios.

Provision a real-byte checkpoint through the standard model redirect. Dummy
weights and simply changing the full checkpoint's layer count are not accepted.
"""

import json
from pathlib import Path

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.pd_utils import run_pd_contract

MODEL = "pd-e2e/DeepSeek-V4-Flash-4layer-W8A8"


@pytest.mark.e2e_model("pd-e2e/DeepSeek-V4-Flash-4layer-W8A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="TP,EP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W8A8_dynamic",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.95, max_wait_seconds=180)
def test_dsv4_pd_compressed_graph_tp2(monkeypatch) -> None:
    # Match the existing DSV4 PD-prefix deployment: retain compressed P
    # checkpoints at page boundaries and do not cache imported KV on D.
    monkeypatch.setenv("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", "4096")
    material = maybe_model_redirect(MODEL)
    assert material != MODEL, "Provision the real DSV4 reduced checkpoint in VLLM_MODEL_REDIRECT_PATH"
    root = Path(material)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["num_hidden_layers"] == 4
    assert config["compress_ratios"] == [0, 0, 4, 128]
    assert config["num_nextn_predict_layers"] == 0
    quantization = json.loads((root / "quant_model_description.json").read_text(encoding="utf-8"))
    assert quantization["model_quant_type"] == "W8A8_DYNAMIC"
    weights = json.loads((root / "quant_model_weights.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    assert not any(name.startswith("mtp.") for name in weights)
    assert all((root / filename).is_file() for filename in set(weights.values()))
    run_pd_contract(
        model=MODEL,
        connector="MooncakeHybridConnector",
        prefix=True,
        tp=2,
        max_num_seqs=1,
        aligned_reference=True,
        compare_logprobs=True,
        decode_prefix=False,
        check_failure=False,
        # A 32-entry C128 page represents 4096 input tokens. Retain two
        # complete compressed pages to exercise prefix reuse and transfer.
        prompt_lengths=(4095, 4096, 4097, 8193),
        max_model_len=12288,
        max_num_batched_tokens=4096,
        block_size=32,
        # Hybrid transfers C128-compressed pages. Two stable A3 runs measured
        # max/mean top-5 deltas of 1.41/0.24 and 1.28/0.26 respectively. Keep
        # mutual greedy top-5 membership and bounded drift, with headroom for
        # the lossy compressed representation.
        distribution_maximum_delta=2.0,
        distribution_mean_delta=0.3,
        extra_args=(
            "--trust-remote-code",
            "--quantization",
            "ascend",
            "--enable-expert-parallel",
            "--tokenizer-mode",
            "deepseek_v4",
            "--kv-cache-memory-bytes",
            "2147483648",
            "--additional-config",
            json.dumps({"enable_cpu_binding": False}),
        ),
    )
