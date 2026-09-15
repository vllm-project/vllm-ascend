# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real MiniMax-M3 language PD across dense/sparse layers and top-k boundary.

This case does not claim multimodal or speculative-decoding coverage.
"""

import json
from pathlib import Path

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.pd_utils import run_pd_contract

MODEL = "pd-e2e/MiniMax-M3-5layer-W8A8"


@pytest.mark.e2e_model("pd-e2e/MiniMax-M3-5layer-W8A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="TP,EP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W8A8_dynamic",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_minimax_pd_sparse_graph_tp2() -> None:
    material = maybe_model_redirect(MODEL)
    assert material != MODEL, "Provision the real MiniMax-M3 reduced checkpoint in VLLM_MODEL_REDIRECT_PATH"
    root = Path(material)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))["text_config"]
    assert config["num_hidden_layers"] == 5
    sparse = config["sparse_attention_config"]
    assert sparse["sparse_attention_freq"] == [0, 0, 0, 1, 1]
    assert sparse["sparse_block_size"] == 128 and sparse["sparse_topk_blocks"] == 16
    index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text(encoding="utf-8"))
    assert all((root / filename).is_file() for filename in set(index["weight_map"].values()))
    quant = json.loads((root / "quant_model_description.json").read_text(encoding="utf-8"))
    assert quant["model_quant_type"] == "W8A8_DYNAMIC"
    run_pd_contract(
        model=MODEL,
        prefix=True,
        tp=2,
        check_failure=False,
        prompt_lengths=(2047, 2048, 2049, 3073),
        extra_args=(
            "--trust-remote-code",
            # The VL class still constructs its vision tower in language-only
            # mode. Select the registered text backbone for this text material.
            "--hf-overrides",
            json.dumps({"architectures": ["MiniMaxM3SparseForCausalLM"]}),
            "--enable-expert-parallel",
            "--quantization",
            "ascend",
            "--max-model-len",
            "4096",
            "--kv-cache-memory-bytes",
            "2147483648",
            "--additional-config",
            json.dumps({"enable_cpu_binding": False}),
        ),
    )
