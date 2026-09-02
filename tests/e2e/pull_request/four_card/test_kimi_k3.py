# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-node A3 (4 logical NPUs) K3 functional tests, not accuracy tests.

Build local configs and initialize dummy weights; no full checkpoint is needed.
Keep production tensor widths with a two-layer, 16-expert target. Cover Block5
at TP4, quantized GQA at DP2/TP2, legacy MLA at P2/D2, and MTP with an image.
Random weights cannot validate checkpoint loading, QuaRot or acceptance rates.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
from PIL import Image
from prometheus_client.parser import text_string_to_metric_families
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import RemoteOpenAIServer, RemotePDServer, VllmRunner

NUM_LAYERS = 2
NUM_EXPERTS = 16
NUM_EXPERTS_PER_TOKEN = 4
NUM_VISION_LAYERS = 1
DRAFT_LAYERS = 1
DRAFT_TARGET_LAYER_IDS = (0,)
VOCAB_SIZE = 163840
MAX_MODEL_LEN = 1024
OUTPUT_TOKENS = 2
# The TP4 K3 state shape makes the aligned hybrid-cache page 1536 tokens.
# Keep the common scenarios at 1024, and enlarge only the prefix-cache engine.
PREFIX_CACHE_MODEL_LEN = 2048
PREFIX_CACHE_TOKENS = 1665
# Keep one KDA and one MLA layer so the smallest target still exercises both
# cache types. Layer zero is the only valid intermediate tap for the drafter.
FULL_ATTN_LAYERS = (2,)
ATTN_RES_BLOCK_SIZE = 2


def _text_config() -> dict:
    return {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "torch_dtype": "bfloat16",
        "hidden_size": 7168,
        "intermediate_size": 33792,
        "num_hidden_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_token": NUM_EXPERTS_PER_TOKEN,
        "num_shared_experts": 2,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "latent_moe_use_norm": True,
        "moe_router_activation_func": "sigmoid",
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_renormalize": True,
        "attn_res_block_size": ATTN_RES_BLOCK_SIZE,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        "num_nextn_predict_layers": 0,
        "linear_attn_config": {
            "head_dim": 128,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "full_attn_layers": list(FULL_ATTN_LAYERS),
            "kda_layers": [i for i in range(1, NUM_LAYERS + 1) if i not in FULL_ATTN_LAYERS],
        },
    }


def _draft_config(variant: str) -> dict:
    config: dict = {
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "hidden_act": "silu",
        "num_hidden_layers": DRAFT_LAYERS,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE,
        "draft_vocab_size": VOCAB_SIZE,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    if variant == "gqa":
        config.update(
            architectures=["DSparkDraftModel"],
            model_type="qwen3",
            # At TP2, 18 / 2 KV heads * 32 dims * K/V matches the
            # target MLA's 512 + 64 compressed-cache dimensions per token.
            num_attention_heads=72,
            num_key_value_heads=18,
            head_dim=32,
            layer_types=["full_attention"] * DRAFT_LAYERS,
            block_size=7,
            num_target_layers=NUM_LAYERS,
            # The drafter consumes intermediate layer outputs, before the final
            # target layer. Keep one tap per reduced draft layer.
            dflash_config={"mask_token_id": 163824, "target_layer_ids": list(DRAFT_TARGET_LAYER_IDS)},
            rope_parameters={
                "rope_type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 65536,
                "rope_theta": 10000.0,
            },
        )
        return config

    config.update(
        architectures=["K3DSparkModel"],
        model_type="k3_dspark",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        target_hidden_size=7168,
        target_num_hidden_layers=NUM_LAYERS,
        num_target_layers=DRAFT_LAYERS,
        target_layer_ids=list(DRAFT_TARGET_LAYER_IDS),
        mask_token_id=163837,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    if variant == "mla_block5":
        config.update(
            num_hidden_layers=DRAFT_LAYERS,
            num_attention_heads=96,
            num_key_value_heads=96,
            head_dim=64,
            qk_head_dim=192,
            num_target_layers=DRAFT_LAYERS,
            target_layer_ids=list(DRAFT_TARGET_LAYER_IDS),
            layer_types=["full_attention"] * DRAFT_LAYERS,
            mask_token_id=163839,
            markov_rank=512,
            sample_from_anchor=True,
            block_size=5,
            full_attention_causal=True,
            dflash_config={"causal": True},
            rope_interleave=True,
        )
        config["rope_parameters"].update(rope_theta=1000000.0, mscale_all_dim=0.0, attn_factor=0.7426255848312643)
    return config


def _write_config(path: Path, config: dict) -> str:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _write_target(path: Path, *, mtp: bool = False) -> str:
    text_config = _text_config()
    text_config["num_nextn_predict_layers"] = int(mtp)
    _write_config(
        path,
        {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "text_config": text_config,
            "vision_config": {"vt_num_hidden_layers": NUM_VISION_LAYERS, "text_hidden_size": 7168},
            "media_placeholder_token_id": 163605,
        },
    )
    # A small local tokenizer/processor description keeps the multimodal wrapper
    # offline too. Token IDs and vocabulary size still match the real K3 model.
    special_tokens = {
        0: "<unk>",
        163584: "<s>",
        163586: "</s>",
        163600: "<|kimi_image_placeholder|>",
        163601: "<|media_begin|>",
        163602: "<|media_content|>",
        163603: "<|media_end|>",
        163605: "<|media_pad|>",
        163839: "<pad>",
    }
    vocabulary = {special_tokens.get(i, f"token_{i}"): i for i in range(VOCAB_SIZE)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=list(special_tokens.values()),
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    ).save_pretrained(path)
    # K3 and K2.5 use the same MoonViT patch format. Reuse vLLM's native image
    # preprocessor rather than copying checkpoint Python code into the test.
    (path / "image_processing_k3_dummy.py").write_text(
        "from vllm.transformers_utils.processors.kimi_k25_vision_fused import KimiK25FusedVisionProcessor\n",
        encoding="utf-8",
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoImageProcessor": "image_processing_k3_dummy.KimiK25FusedVisionProcessor"},
                "media_proc_cfg": {
                    "patch_size": 14,
                    "merge_kernel_size": 2,
                    "temporal_merge_kernel_size": 4,
                    "in_patch_limit": 256,
                    "patch_limit_on_one_side": 16,
                    "fixed_output_tokens": None,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _write_w4a8_description(path: Path) -> None:
    # Use the released mixed KDA precision: W8A8 q/k/v, floating-point gates;
    # routed experts use W4A8. No rotation file means this is NOT a QuaRot test.
    description = {"model.embed_tokens.weight": "FLOAT", "lm_head.weight": "FLOAT", "optional": {}}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"
        for projection in ("self_attention_res_proj", "mlp_res_proj"):
            description[f"{prefix}.{projection}.weight"] = "FLOAT"
        if layer + 1 in FULL_ATTN_LAYERS:
            quantized = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa")
            floating: tuple[str, ...] = ("kv_b_proj", "o_proj", "g_proj")
        else:
            quantized = ("q_proj", "k_proj", "v_proj")
            floating = ("g_proj", "f_a_proj", "f_b_proj", "b_proj", "o_proj", "q_conv1d", "k_conv1d", "v_conv1d")
        for projection in quantized:
            description[f"{prefix}.self_attn.{projection}.weight"] = "W8A8_DYNAMIC"
        for projection in floating:
            description[f"{prefix}.self_attn.{projection}.weight"] = "FLOAT"
        mlp_prefix = f"{prefix}.block_sparse_moe.shared_experts" if layer else f"{prefix}.mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            description[f"{mlp_prefix}.{projection}.weight"] = "W8A8_DYNAMIC"
        if layer:
            for projection in ("routed_expert_down_proj", "routed_expert_up_proj"):
                description[f"{prefix}.block_sparse_moe.{projection}.weight"] = "W8A8_DYNAMIC"
            # ModelSlim selects the fused expert group's scheme from expert 0.
            # All 16 experts still run; duplicating their identical descriptions
            # only bloats the config sent to each spawned worker.
            for projection in ("w1", "w2", "w3"):
                description[f"{prefix}.block_sparse_moe.experts.0.{projection}.weight"] = "W4A8_DYNAMIC"
    description = {f"language_model.{key}" if key != "optional" else key: value for key, value in description.items()}
    for layer in range(NUM_VISION_LAYERS):
        for projection in ("mlp.fc0", "mlp.fc1", "wqkv", "wo"):
            description[f"vision_tower.encoder.blocks.{layer}.{projection}.weight"] = "FLOAT"
    for projection in ("proj.0", "proj.2", "rot_proj"):
        description[f"mm_projector.{projection}.weight"] = "FLOAT"
    (path / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")


@pytest.fixture(scope="module")
def k3_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_path = tmp_path_factory.mktemp("k3-dummy")
    models = {"target": _write_target(tmp_path / "target")}
    models["w4a8"] = _write_target(tmp_path / "w4a8")
    _write_w4a8_description(tmp_path / "w4a8")
    models["mtp"] = _write_target(tmp_path / "mtp", mtp=True)
    for variant in ("mla", "mla_block5", "gqa"):
        models[variant] = _write_config(tmp_path / variant, _draft_config(variant))
    return models


@pytest.fixture(autouse=True)
def k3_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")


def _engine_args(models: dict[str, str], variant: str, tp: int = 4) -> dict:
    steps = 5 if variant == "mla_block5" else 7
    # Respect upstream LCM(TP, speculative query width), including Block5's 48.
    graph_sizes = [48, 96] if steps == 5 else [16, 32]
    return {
        "load_format": "dummy",
        "dtype": "bfloat16",
        "tensor_parallel_size": tp,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 512,
        "block_size": 128,
        "kv_cache_memory_bytes": 512 * 1024**2,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "mamba_cache_mode": "align",
        "async_scheduling": True,
        "disable_log_stats": False,
        "seed": 0,
        "limit_mm_per_prompt": {"image": 0},
        "mm_encoder_tp_mode": "data",
        "speculative_config": {
            "method": "dspark",
            "model": models[variant],
            "num_speculative_tokens": steps,
            "draft_sample_method": "greedy",
            "enforce_eager": True,
            "draft_load_config": {"load_format": "dummy"},
        },
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": graph_sizes},
        "additional_config": {
            "enable_cpu_binding": False,
            "enable_shared_expert_dp": False,
            "multistream_overlap_shared_expert": True,
            "enable_fused_mc2": 0,
        },
    }


def _prompt(length: int, salt: int = 0) -> dict:
    return {"prompt_token_ids": [10 + (i + salt) % 1000 for i in range(length)]}


def _generate(llm, prompts: list[dict]):
    print(f"K3 smoke: generating {OUTPUT_TOKENS} tokens for {len(prompts)} requests", flush=True)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.finished
        assert len(output.outputs) == 1
        assert len(output.outputs[0].token_ids) == OUTPUT_TOKENS
        assert all(0 <= token < VOCAB_SIZE for token in output.outputs[0].token_ids)
    print(f"K3 smoke completed; cached tokens: {[output.num_cached_tokens for output in outputs]}", flush=True)
    return outputs


def test_k3_mla_block5_tp4(k3_models: dict[str, str]) -> None:
    args = _engine_args(k3_models, "mla_block5")
    args["max_model_len"] = PREFIX_CACHE_MODEL_LEN
    with VllmRunner(k3_models["target"], **args) as runner:
        llm = runner.model
        # Kernel-block boundaries in one mixed-length wave.
        _generate(llm, [_prompt(length, salt=i * 137) for i, length in enumerate((127, 128, 129))])

        # Cross the aligned 1536-token hybrid-cache page. This also exercises
        # chunked prefill and the high block-table columns, so no duplicate
        # near-max-length wave is needed in the reduced test.
        prefix = _prompt(PREFIX_CACHE_TOKENS, salt=421)
        assert llm.reset_prefix_cache()
        created = _generate(llm, [prefix])[0]
        assert created.num_cached_tokens == 0
        assert created.num_cache_creation_tokens > 0, "Request did not create an aligned hybrid-cache page"

        drafts = [m for m in llm.get_metrics() if m.name == "vllm:spec_decode_num_drafts"]
        assert drafts and all(isinstance(m, Counter) for m in drafts)
        assert sum(m.value for m in drafts) > 0, "Requests bypassed speculative decoding"


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_k3_gqa_dcp_replicated_draft_kv(k3_models: dict[str, str], dcp_size: int) -> None:
    args = _engine_args(k3_models, "gqa")
    args["decode_context_parallel_size"] = dcp_size
    args["enforce_eager"] = True
    args.pop("compilation_config")
    with VllmRunner(k3_models["target"], **args) as runner:
        llm = runner.model
        # Exercise the first token, block boundaries, chunked prefill, and the
        # last block-table columns while the target cache is DCP-sharded and
        # every draft-DCP lane addresses a replicated physical cache range.
        _generate(llm, [_prompt(length, salt=i * 137) for i, length in enumerate((1, 127, 128, 129, 769))])
        _generate(llm, [_prompt(MAX_MODEL_LEN - OUTPUT_TOKENS, salt=911)])

        prefix = _prompt(1153, salt=421)
        assert llm.reset_prefix_cache()
        cold = _generate(llm, [prefix])[0]
        warm = _generate(llm, [prefix])[0]
        assert cold.num_cached_tokens == 0
        assert warm.num_cached_tokens > 0, "Repeated prompt did not reuse the hybrid prefix cache"

        drafts = [m for m in llm.get_metrics() if m.name == "vllm:spec_decode_num_drafts"]
        assert drafts and all(isinstance(m, Counter) for m in drafts)
        assert sum(m.value for m in drafts) > 0, "Requests bypassed speculative decoding"


def _serve_args(args: dict) -> list[str]:
    result = [
        "--served-model-name",
        "k3-dummy",
        "--host",
        "127.0.0.1",
        "--trust-remote-code",
        "--enable-prompt-tokens-details",
    ]
    for name, value in args.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(option)
        else:
            result.extend([option, json.dumps(value) if isinstance(value, dict) else str(value)])
    return result


def _completion(url: str, prompt: list[int], *, max_tokens: int = OUTPUT_TOKENS, **kwargs) -> dict:
    response = requests.post(
        url,
        json={
            "model": "k3-dummy",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "return_token_ids": True,
            **kwargs,
        },
        timeout=180,
    )
    response.raise_for_status()
    output = response.json()
    assert output["usage"]["completion_tokens"] == max_tokens
    assert len(output["choices"][0]["token_ids"]) == max_tokens
    assert all(0 <= token < VOCAB_SIZE for token in output["choices"][0]["token_ids"])
    return output


def _draft_counts(url: str) -> dict[str, float]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return {
        sample.labels["engine"]: sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "vllm:spec_decode_num_drafts_total"
    }


def test_k3_gqa_w4a8_dp2_tp2(k3_models: dict[str, str]) -> None:
    args = _engine_args(k3_models, "gqa", tp=2)
    args["data_parallel_size"] = 2
    args["quantization"] = "ascend"
    args["additional_config"]["enable_shared_expert_dp"] = True
    port = get_open_port()
    with RemoteOpenAIServer(
        k3_models["w4a8"],
        [*_serve_args(args), "--port", str(port), "--api-server-count", "1"],
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        max_wait_seconds=600,
    ) as server:
        # Four concurrent requests ensure both DP engines execute; 769 also
        # crosses the 512-token chunked-prefill boundary.
        lengths = (1, 127, 128, 769)
        prompts = [_prompt(length, salt=i * 137)["prompt_token_ids"] for i, length in enumerate(lengths)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda prompt: _completion(server.url_for("v1", "completions"), prompt), prompts))
        assert len(outputs) == len(prompts)
        counts = _draft_counts(server.url_for("metrics"))
        assert len(counts) == 2 and all(count > 0 for count in counts.values()), counts


def test_k3_mtp_image_tp4(k3_models: dict[str, str]) -> None:
    args = _engine_args(k3_models, "gqa")
    args["speculative_config"] = {
        "method": "mtp",
        "num_speculative_tokens": 1,
        "enforce_eager": True,
        "draft_load_config": {"load_format": "dummy"},
    }
    args["limit_mm_per_prompt"] = {"image": 1}
    with VllmRunner(k3_models["mtp"], **args) as runner:
        image_prompt = {
            "prompt": "<|kimi_image_placeholder|> describe",
            "multi_modal_data": {"image": Image.new("RGB", (56, 56), (32, 64, 128))},
        }
        _generate(runner.model, [image_prompt])
        drafts = [m for m in runner.model.get_metrics() if m.name == "vllm:spec_decode_num_drafts"]
        assert drafts and sum(m.value for m in drafts) > 0, "Requests bypassed MTP"


def test_k3_mla_pd_tp2(k3_models: dict[str, str]) -> None:
    prefill_port, decode_port = get_open_port(), get_open_port()
    transfer_config = {
        "kv_connector": "MooncakeConnectorV1",
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 2},
            "decode": {"dp_size": 1, "tp_size": 2},
        },
    }
    prefill_args = _engine_args(k3_models, "mla", tp=2)
    # P also builds the draft KV that D consumes; both peers need the same
    # target/draft layer layout even though P only generates one token.
    prefill_args.pop("compilation_config")
    prefill_args["enforce_eager"] = True
    prefill_args["kv_transfer_config"] = dict(transfer_config, kv_role="kv_producer", kv_port=get_open_port())
    decode_args = _engine_args(k3_models, "mla", tp=2)
    decode_args["kv_transfer_config"] = dict(transfer_config, kv_role="kv_consumer", kv_port=get_open_port())
    servers = [
        [k3_models["target"], "--port", str(prefill_port), *_serve_args(prefill_args)],
        [k3_models["target"], "--port", str(decode_port), *_serve_args(decode_args)],
    ]
    # Use the normal P/D transfer protocol directly so missing transfer metadata
    # cannot silently fall back to local prefill and make this test pass.
    with RemotePDServer(servers):
        prefill_url = f"http://127.0.0.1:{prefill_port}/v1/completions"
        decode_url = f"http://127.0.0.1:{decode_port}/v1/completions"
        prompt = _prompt(129, salt=129)["prompt_token_ids"]
        prefill = _completion(
            prefill_url,
            prompt,
            max_tokens=1,
            kv_transfer_params={"do_remote_decode": True, "do_remote_prefill": False},
        )
        transfer = prefill["kv_transfer_params"]
        assert transfer["do_remote_prefill"]
        assert any(transfer["remote_block_ids"])
        decoded = _completion(decode_url, prompt, kv_transfer_params=transfer)
        assert decoded["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        # A D worker can also receive a request without remote KV. Its MLA
        # prefill weights must remain usable (the previous P/D fallback bug).
        _completion(decode_url, _prompt(257, salt=911)["prompt_token_ids"])
        counts = _draft_counts(f"http://127.0.0.1:{decode_port}/metrics")
        assert counts and sum(counts.values()) > 0
