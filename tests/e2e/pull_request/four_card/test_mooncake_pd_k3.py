# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 hybrid-cache PD contract using a provisioned reduced checkpoint.

The material ID is resolved through VLLM_MODEL_REDIRECT_PATH, not downloaded
as a guessed public repository. Missing material is a failure, never a pass.
The reduced checkpoint must preserve three KDA layers and one MLA layer.
"""

import json
from pathlib import Path

import pytest
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.pd_utils import run_pd_contract

MODEL = "pd-e2e/Kimi-K3-Text-4layer-16expert-W4A8"


@pytest.mark.e2e_model("pd-e2e/Kimi-K3-Text-4layer-16expert-W4A8")
@pytest.mark.e2e_coverage(
    arch="mamba_ssm",
    feature="aclgraph,prefix_caching,chunked_prefill,mixed_lengths",
    parallel="TP,EP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_k3_hybrid_pd_prefix_graph_tp2() -> None:
    material = maybe_model_redirect(MODEL)
    assert material != MODEL, "Provision the K3 reduced checkpoint in VLLM_MODEL_REDIRECT_PATH"
    config = json.loads((Path(material) / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["KimiK3ForCausalLM"]
    text_config = config
    assert config["model_type"] == "kimi_linear"
    assert text_config["num_hidden_layers"] == 4
    assert text_config["num_experts"] == 16
    assert text_config["linear_attn_config"]["kda_layers"] == [1, 2, 3]
    assert text_config["linear_attn_config"]["full_attn_layers"] == [4]
    assert list(Path(material).glob("*.safetensors")), "Real checkpoint shards are required"
    weights = json.loads((Path(material) / "quant_model_weights.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    assert not any(name.endswith((".experts.gate_up_proj", ".experts.down_proj")) for name in weights), (
        "Export training-packed experts to the checkpoint's external w1/w2/w3 format before running E2E"
    )
    for layer in (1, 2, 3):
        for expert in range(text_config["num_experts"]):
            for projection in ("w1", "w2", "w3"):
                name = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"
                assert name in weights, f"Missing real expert weight: {name}"
    tokenizer_config = json.loads((Path(material) / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_tokens = tokenizer_config["added_tokens_decoder"]
    assert special_tokens[str(text_config["bos_token_id"])]["content"] == tokenizer_config["bos_token"]
    assert special_tokens[str(text_config["pad_token_id"])]["content"] == tokenizer_config["pad_token"]
    assert special_tokens[str(text_config["eos_token_id"])]["content"] == "<|end_of_msg|>"
    assert (Path(material) / "tiktoken.model").is_file(), "Real K3 tokenizer vocabulary is required"
    run_pd_contract(
        model=MODEL,
        prefix=True,
        tp=2,
        max_num_seqs=1,
        aligned_reference=True,
        check_failure=False,
        # The actual TP2 hybrid runtime aligns its attention page to 3072
        # tokens. Cross that boundary and retain two full pages for prefix reuse.
        prompt_lengths=(3071, 3072, 3073, 6145),
        extra_args=(
            "--trust-remote-code",
            "--quantization",
            "ascend",
            "--enable-expert-parallel",
            "--mamba-cache-mode",
            "align",
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            "3072",
            "--kv-cache-memory-bytes",
            "1073741824",
            "--additional-config",
            json.dumps({"enable_cpu_binding": False}),
        ),
    )
