# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GLM5.2 PD: shared indexers and the 2048-token top-k boundary.

The A3 material uses W8A8 linear projections and W8A8_DYNAMIC MoE experts;
the model-level quantization label alone does not describe its expert path.
"""

import json
from pathlib import Path

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.pd_utils import run_pd_contract

MODEL = "pd-e2e/GLM-5.2-7layer-W8A8"


@pytest.mark.e2e_model("pd-e2e/GLM-5.2-7layer-W8A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="TP,EP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W8A8,W8A8_dynamic",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_glm_pd_shared_indexer_graph_tp2() -> None:
    material = maybe_model_redirect(MODEL)
    assert material != MODEL, "Provision the real GLM5.2 reduced checkpoint in VLLM_MODEL_REDIRECT_PATH"
    root = Path(material)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["num_hidden_layers"] == 7 and config["num_nextn_predict_layers"] == 0
    assert config["indexer_types"] == ["full", "full", "full", "shared", "shared", "shared", "full"]
    assert len(config["mlp_layer_types"]) == 7
    assert config["index_topk"] == 2048
    weights = json.loads((root / "quant_model_weights.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    assert not any(name.startswith("model.layers.7.") for name in weights)
    assert all((root / filename).is_file() for filename in set(weights.values()))
    quantization = json.loads((root / "quant_model_description.json").read_text(encoding="utf-8"))
    assert quantization["model_quant_type"] == "W8A8"
    assert any(isinstance(value, str) and value == "W8A8_DYNAMIC" for value in quantization.values())
    run_pd_contract(
        model=MODEL,
        prefix=True,
        tp=2,
        compare_logprobs=True,
        check_failure=False,
        prompt_lengths=(2047, 2048, 2049, 3073),
        extra_args=(
            "--trust-remote-code",
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
