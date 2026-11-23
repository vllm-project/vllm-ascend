# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, VocabParallelMode, VocabParallelPlan
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.lm_head import select_lm_head_method
from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8DynamicLmHeadMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
    AscendW8A8MXFP8DynamicLmHeadMethod,
)


def runtime(role="generate", enabled=True, tied=False):
    return SimpleNamespace(
        use_v2_model_runner=True,
        quant_config=None,
        additional_config={"lm_head_quantization": "w8a8_dynamic"} if enabled else {},
        model_config=SimpleNamespace(
            runner_type=role,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            hf_text_config=SimpleNamespace(tie_word_embeddings=tied),
        ),
    )


@pytest.fixture(autouse=True)
def standard_device():
    with patch(
        "vllm_ascend.quantization.lm_head.get_current_hardware_profile",
        return_value=SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.STANDARD),
    ):
        yield


@pytest.fixture
def a5_device():
    with (
        patch(
            "vllm_ascend.quantization.lm_head.get_current_hardware_profile",
            return_value=SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED),
        ),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config", return_value=runtime()),
    ):
        yield


def test_config_accepts_only_target_enum():
    assert AscendConfig(sparse_kv_offload_config=None).lm_head_quantization is None
    assert (
        AscendConfig(sparse_kv_offload_config=None, lm_head_quantization="w8a8_dynamic").lm_head_quantization
        == "w8a8_dynamic"
    )
    for value in ("auto", "W8A8_DYNAMIC", True):
        with pytest.raises(ValueError):
            AscendConfig(sparse_kv_offload_config=None, lm_head_quantization=value)
    with pytest.raises(ValueError):
        AscendConfig(sparse_kv_offload_config=None, draft_lm_head_quantization="w8a8_dynamic")


def test_lm_head_option_does_not_restrict_other_runtime_settings():
    config = AscendConfig(
        sparse_kv_offload_config=SparseKVOffloadConfig(),
        lm_head_quantization="w8a8_dynamic",
        finegrained_tp_config={"lmhead_tensor_parallel_size": 2},
        rl_config={"enabled": True},
    )
    vc = runtime()
    vc.model_config.is_moe = False
    vc.model_config.is_deepseek_mla = False
    vc.model_config.enforce_eager = True
    vc.lora_config = object()
    vc.weight_transfer_config = object()
    vc.offload_config = SimpleNamespace(
        uva=SimpleNamespace(cpu_offload_gb=1),
        prefetch=SimpleNamespace(offload_group_size=1),
    )
    vc.use_v2_model_runner = True
    vc.parallel_config = SimpleNamespace(
        pipeline_parallel_size=2,
        prefill_context_parallel_size=2,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_batch_sharded_sampling=False,
    )
    vc.load_config = SimpleNamespace(load_format="sharded_state")
    vc.speculative_config = SimpleNamespace(method="eagle")
    vc.kv_transfer_config = None
    vc.cache_config = SimpleNamespace(cache_dtype="auto")
    vc.attention_config = SimpleNamespace(indexer_kv_dtype="auto")
    with (
        patch("vllm_ascend.utils.enable_sp", return_value=False),
        patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=False),
        patch("vllm_ascend.utils.model_uses_kpool_indexer", return_value=False),
    ):
        assert config.derive_and_validate(vc) is config


def test_float_target_and_role_isolation():
    head = torch.nn.Module()
    original = UnquantizedEmbeddingMethod()
    assert select_lm_head_method(head, "lm_head", None, original, runtime(enabled=False)) is original
    assert select_lm_head_method(head, "lm_head", None, original, runtime(role="draft")) is original
    assert select_lm_head_method(head, "model.markov_w2", None, original, runtime()) is original
    selected = select_lm_head_method(head, "language_model.lm_head", None, original, runtime())
    assert type(selected) is AscendEmbeddingMethod
    assert isinstance(selected.quant_method, AscendW8A8DynamicLmHeadMethod)
    with pytest.raises(ValueError, match="share embedding"):
        select_lm_head_method(None, "lm_head", None, original, runtime(tied=True))


@pytest.mark.parametrize("prefix", ["lm_head", "language_model.lm_head", "head", "model.head"])
@pytest.mark.parametrize("role", ["generate", "draft"])
def test_output_head_names_and_draft_isolation(prefix, role):
    original = UnquantizedEmbeddingMethod()
    selected = select_lm_head_method(torch.nn.Module(), prefix, None, original, runtime(role=role))
    if role == "draft":
        assert selected is original
    else:
        assert type(selected) is AscendEmbeddingMethod
        assert isinstance(selected.quant_method, AscendW8A8DynamicLmHeadMethod)


@pytest.mark.parametrize("hf_tied", [False, True, None, "missing"])
@pytest.mark.parametrize("text_tied", [False, True, None, "missing"])
def test_tied_embeddings_checks_both_model_configs(hf_tied, text_tied):
    config = runtime()
    for name, tied in (("hf_config", hf_tied), ("hf_text_config", text_tied)):
        if tied == "missing":
            delattr(config.model_config, name)
        else:
            setattr(config.model_config, name, None if tied is None else SimpleNamespace(tie_word_embeddings=tied))
    original = UnquantizedEmbeddingMethod()
    if hf_tied is True or text_tied is True:
        with pytest.raises(ValueError, match="share embedding"):
            select_lm_head_method(None, "lm_head", None, original, config)
    else:
        selected = select_lm_head_method(torch.nn.Module(), "lm_head", None, original, config)
        assert isinstance(selected.quant_method, AscendW8A8DynamicLmHeadMethod)


@pytest.mark.parametrize("recipe", ["FLOAT", "W8A8_DYNAMIC", "W8A8", None])
def test_modelslim_checkpoint_recipe(recipe):
    head = torch.nn.Module()
    description = {} if recipe is None else {"lm_head.weight": recipe}
    config = AscendModelSlimConfig(description)
    original = (
        AscendEmbeddingMethod(AscendW8A8DynamicLinearMethod())
        if recipe == "W8A8_DYNAMIC"
        else UnquantizedEmbeddingMethod()
    )
    if recipe in ("W8A8", None):
        with pytest.raises(ValueError):
            select_lm_head_method(None, "lm_head", config, original, runtime())
    else:
        selected = select_lm_head_method(head, "lm_head", config, original, runtime())
        if recipe == "FLOAT":
            assert type(selected) is AscendEmbeddingMethod
        else:
            assert selected is original


def test_compressed_tensors_preserves_checkpoint_quantization():
    config = AscendCompressedTensorsConfig(target_scheme_map={"Linear": {}}, ignore=["lm_head"], quant_format="")
    selected = select_lm_head_method(torch.nn.Module(), "lm_head", config, None, runtime())
    assert type(selected) is AscendEmbeddingMethod
    config.ignore = []
    original = AscendEmbeddingMethod(AscendW8A8DynamicLinearMethod())
    assert select_lm_head_method(None, "lm_head", config, original, runtime()) is original
    assert select_lm_head_method(None, "lm_head", config, None, runtime()) is None


def test_compressed_tensors_uses_format_contract():
    config = SimpleNamespace(
        get_name=lambda: "compressed-tensors",
        ignore=["lm_head"],
        packed_modules_mapping={},
    )
    selected = select_lm_head_method(torch.nn.Module(), "lm_head", config, UnquantizedEmbeddingMethod(), runtime())
    assert type(selected) is AscendEmbeddingMethod


def test_modelslim_preserves_registered_checkpoint_method():
    class SpecializedDynamicMethod(AscendW8A8DynamicLinearMethod):
        pass

    config = AscendModelSlimConfig({"lm_head.weight": "W8A8_DYNAMIC"})
    original = AscendEmbeddingMethod(SpecializedDynamicMethod())
    assert select_lm_head_method(None, "lm_head", config, original, runtime()) is original


def test_lm_head_scheme_dispatch_is_extensible():
    class AlternativeOnlineMethod(AscendW8A8DynamicLmHeadMethod):
        pass

    config = runtime()
    config.additional_config["lm_head_quantization"] = "alternative_dynamic"
    with patch.dict(
        "vllm_ascend.quantization.methods.registry._SCHEME_REGISTRY",
        {("ALTERNATIVE_DYNAMIC", "lm_head"): AlternativeOnlineMethod},
    ):
        selected = select_lm_head_method(torch.nn.Module(), "lm_head", None, UnquantizedEmbeddingMethod(), config)
        assert type(selected) is AscendEmbeddingMethod
        assert type(selected.quant_method) is AlternativeOnlineMethod
        checkpoint_config = AscendModelSlimConfig({"lm_head.weight": "ALTERNATIVE_DYNAMIC"})
        original = AscendEmbeddingMethod(AlternativeOnlineMethod())
        assert select_lm_head_method(None, "lm_head", checkpoint_config, original, config) is original


def test_unregistered_lm_head_scheme_fails():
    config = runtime()
    config.additional_config["lm_head_quantization"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported LM head quantization"):
        select_lm_head_method(None, "lm_head", None, UnquantizedEmbeddingMethod(), config)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
def test_online_lm_head_loads_bias_with_standard_loader(dtype, rank):
    plan = VocabParallelPlan(VocabParallelMode.STANDARD, rank, 2)
    with (
        patch("vllm_ascend.ops.vocab_parallel_embedding._resolve_vocab_parallel_plan", return_value=plan),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config_or_none", return_value=runtime()),
    ):
        head = AscendParallelLMHead(5, 4, bias=True, params_dtype=dtype, padding_size=2, prefix="lm_head")

    weight = torch.arange(20, dtype=dtype).view(5, 4)
    bias = torch.arange(5, dtype=dtype)
    head.weight.weight_loader(head.weight, weight)
    head.bias.weight_loader(head.bias, bias)
    start = rank * 3
    rows = min(3, 5 - start)
    assert type(head.quant_method) is AscendEmbeddingMethod
    assert head.bias.dtype == dtype
    torch.testing.assert_close(head.weight[:rows], weight[start : start + rows])
    torch.testing.assert_close(head.bias[:rows], bias[start : start + rows])
    assert torch.count_nonzero(head.weight[rows:]) == 0
    assert torch.count_nonzero(head.bias[rows:]) == 0
    assert head.weight.weight_loader == head.weight_loader
    assert head.bias.weight_loader == head.weight_loader

    x = torch.ones(2, 4, dtype=dtype)
    with (
        patch("torch_npu.npu_dynamic_quant", return_value=(x.to(torch.int8), torch.ones(2))),
        patch("torch_npu.npu_quant_matmul", return_value=torch.empty(2, 3, dtype=dtype)) as matmul,
    ):
        head.quant_method.apply(head, x, bias=head.bias)
    assert matmul.call_args.kwargs["bias"] is head.bias


def test_online_quantization_processes_loaded_shard():
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float16).view(4, 3), requires_grad=False)
    head.weight_scale = torch.nn.Parameter(torch.empty(4, 1), requires_grad=False)
    head.weight_offset = torch.nn.Parameter(torch.empty(4, 1), requires_grad=False)

    def fake_dynamic_quant(weight, dst_type):
        assert dst_type == torch.int8
        return weight.to(torch.int8), torch.ones(weight.shape[0], dtype=torch.float32)

    with (
        patch("torch_npu.npu_dynamic_quant", side_effect=fake_dynamic_quant) as quant,
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.maybe_trans_nz", side_effect=lambda x: x),
    ):
        method = AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())
        method.process_weights_after_loading(head)
        method.process_weights_after_loading(head)
    assert quant.call_count == 1
    assert head.weight.dtype == torch.int8
    assert head.weight.shape == (3, 4)
    assert torch.equal(head.weight_scale, torch.ones(4))
    assert torch.equal(head.weight_offset, torch.zeros(4))


def test_v1_cannot_enable_target_lm_head_quantization():
    config = runtime()
    config.use_v2_model_runner = False
    original = UnquantizedEmbeddingMethod()
    with pytest.raises(ValueError, match="Model Runner V2"):
        select_lm_head_method(None, "lm_head", None, original, config)
    config.additional_config = {}
    assert select_lm_head_method(None, "lm_head", None, original, config) is original


@pytest.mark.parametrize("format_name", ["float", "modelslim", "compressed-tensors"])
def test_a5_selects_online_mxfp8_for_float_head(a5_device, format_name):
    configs = {
        "float": None,
        "modelslim": AscendModelSlimConfig({"lm_head.weight": "FLOAT"}),
        "compressed-tensors": AscendCompressedTensorsConfig({"Linear": {}}, ["lm_head"], ""),
    }
    selected = select_lm_head_method(None, "lm_head", configs[format_name], UnquantizedEmbeddingMethod(), runtime())
    assert type(selected) is AscendEmbeddingMethod
    assert type(selected.quant_method) is AscendW8A8MXFP8DynamicLmHeadMethod
    assert selected.quant_method.group_size == 32


@pytest.mark.parametrize("recipe", ["W8A8_DYNAMIC", "W8A8_MXFP8"])
def test_a5_preserves_checkpoint_method(a5_device, recipe):
    scheme = AscendW8A8MXFP8DynamicLinearMethod() if recipe == "W8A8_MXFP8" else AscendW8A8DynamicLinearMethod()
    original = AscendEmbeddingMethod(scheme)
    config = AscendModelSlimConfig({"lm_head.weight": recipe})
    assert select_lm_head_method(None, "lm_head", config, original, runtime()) is original


@pytest.mark.parametrize(
    "config,prefix",
    [(runtime(role="draft"), "lm_head"), (runtime(enabled=False), "lm_head"), (runtime(), "embed_tokens")],
)
def test_a5_preserves_unselected_heads(a5_device, config, prefix):
    original = UnquantizedEmbeddingMethod()
    assert select_lm_head_method(None, prefix, None, original, config) is original


def test_a5_dispatch_supports_registered_schemes(a5_device):
    class AlternativeOnlineMethod(AscendW8A8DynamicLmHeadMethod):
        a5_quant_type = "ALTERNATIVE_MXFP8"

    class AlternativeMXFP8Method(AscendW8A8MXFP8DynamicLmHeadMethod):
        pass

    config = runtime()
    config.additional_config["lm_head_quantization"] = "alternative_dynamic"
    with patch.dict(
        "vllm_ascend.quantization.methods.registry._SCHEME_REGISTRY",
        {
            ("ALTERNATIVE_DYNAMIC", "lm_head"): AlternativeOnlineMethod,
            ("ALTERNATIVE_MXFP8", "lm_head"): AlternativeMXFP8Method,
        },
    ):
        selected = select_lm_head_method(None, "lm_head", None, UnquantizedEmbeddingMethod(), config)
        assert type(selected.quant_method) is AlternativeMXFP8Method


def test_a5_online_mx_settings_do_not_inherit_checkpoint_group_size(a5_device):
    config = runtime()
    config.quant_config = SimpleNamespace(quant_description={"group_size": 128})
    with (
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=1),
    ):
        method = AscendW8A8MXFP8DynamicLmHeadMethod()
    assert method.group_size == 32
    assert method.dynamic_mx_quant_scale_alg == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("hidden_size", [64, 96])
def test_a5_online_quantization_loads_and_transforms_shards(a5_device, dtype, rank, hidden_size):
    plan = VocabParallelPlan(VocabParallelMode.STANDARD, rank, 2)
    with (
        patch("vllm_ascend.ops.vocab_parallel_embedding._resolve_vocab_parallel_plan", return_value=plan),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_current_vllm_config_or_none", return_value=runtime()),
    ):
        head = AscendParallelLMHead(8301, hidden_size, bias=True, params_dtype=dtype, prefix="lm_head")
    weight = (torch.arange(8301 * hidden_size).reshape(8301, hidden_size) % 8).to(dtype)
    bias = (torch.arange(8301) % 8).to(dtype)
    head.weight.weight_loader(head.weight, weight)
    head.bias.weight_loader(head.bias, bias)
    local_weight = head.weight.detach().clone()
    local_bias = head.bias.detach().clone()
    groups = (hidden_size + 31) // 32
    scale_pairs = (groups + 1) // 2

    def fake_mx_quant(value, *, dst_type, block_size=32, scale_alg):
        assert dst_type == torch.float8_e4m3fn
        assert block_size == 32
        assert scale_alg == head.quant_method.quant_method.dynamic_mx_quant_scale_alg
        scale = (torch.arange(value.shape[0] * scale_pairs * 2) % 8 + 120).to(torch.uint8)
        scale = scale.reshape(value.shape[0], scale_pairs, 2)
        if groups % 2:
            scale[:, -1, -1] = 0
        return value.to(dst_type), scale

    expected_scale = fake_mx_quant(
        local_weight, dst_type=torch.float8_e4m3fn, scale_alg=head.quant_method.quant_method.dynamic_mx_quant_scale_alg
    )[1]
    with (
        patch("torch_npu.npu_dynamic_mx_quant", side_effect=fake_mx_quant) as quant,
        patch("torch_npu.npu_dynamic_quant") as int8_quant,
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.maybe_trans_nz", side_effect=lambda value, **_: value),
    ):
        head.quant_method.process_weights_after_loading(head)
        head.quant_method.process_weights_after_loading(head)
    assert quant.call_count == 2
    int8_quant.assert_not_called()
    assert head.weight.dtype == torch.float8_e4m3fn
    assert head.weight_scale.dtype == torch.uint8
    assert head.weight.shape == (hidden_size, 4160)
    assert head.weight_scale.shape == (scale_pairs, 4160, 2)
    torch.testing.assert_close(head.weight.float(), local_weight.T.float())
    torch.testing.assert_close(head.weight_scale, expected_scale.transpose(0, 1))
    torch.testing.assert_close(head.bias, local_bias)
    assert head.weight.weight_loader == head.weight_loader
    assert not hasattr(head, "weight_offset")
    if rank == 1:
        assert torch.count_nonzero(head.weight.float()[:, 4141:]) == 0

    x = torch.ones(2, hidden_size, dtype=dtype)
    with (
        patch("torch_npu.npu_dynamic_mx_quant", side_effect=fake_mx_quant),
        patch("torch_npu.npu_quant_matmul", return_value=torch.empty(2, 4160, dtype=dtype)) as matmul,
    ):
        output = head.quant_method.apply(head, x, bias=head.bias)
    assert output.dtype == dtype
    kwargs = matmul.call_args.kwargs
    assert matmul.call_args.args[0].dtype == torch.float8_e4m3fn
    assert kwargs["scale_dtype"] == torch_npu.float8_e8m0fnu
    assert kwargs["pertoken_scale_dtype"] == torch_npu.float8_e8m0fnu
    assert kwargs["group_sizes"] == [1, 1, 32]
    assert kwargs["output_dtype"] == dtype
    torch.testing.assert_close(kwargs["bias"], local_bias.float())


def test_a5_online_quantization_requires_float_weights(a5_device):
    method = AscendW8A8MXFP8DynamicLmHeadMethod()
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        method.get_weight(64, 128, torch.float32)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(128, 64, dtype=torch.int8), requires_grad=False)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        method.process_weights_after_loading(layer)
