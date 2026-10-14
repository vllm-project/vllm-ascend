# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.utils import supports_non_contiguous_kv_cache


def _config(architecture, layer_types=None, **text_options):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            use_mla=False,
            hf_config=SimpleNamespace(architectures=[architecture]),
            hf_text_config=SimpleNamespace(layer_types=layer_types, **text_options),
        )
    )


@pytest.mark.parametrize(
    "architecture",
    ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "Qwen3_5MTP", "Qwen3_5MoeMTP"],
)
def test_supported_dense_model_scope(architecture):
    assert supports_non_contiguous_kv_cache(_config(architecture))
    assert supports_non_contiguous_kv_cache(_config(architecture, ["full_attention"] * 4))


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
@pytest.mark.parametrize("layer_types", [["linear_attention", "full_attention"], ["full_attention"]])
def test_supported_hybrid_model_scope(architecture, layer_types):
    assert supports_non_contiguous_kv_cache(_config(architecture, layer_types))


@pytest.mark.parametrize(
    "architecture",
    [
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
        "Glm5NextForCausalLM",
        "DeepseekV3ForCausalLM",
        "MambaForCausalLM",
        "JambaForCausalLM",
        "UnknownForCausalLM",
        "Qwen3ForCausalLMLocal",
        None,
    ],
)
def test_other_model_families_keep_legacy_scope(architecture):
    config = _config(architecture, ["full_attention", "linear_attention"])
    config.model_config.is_hybrid = False
    config.model_config.hf_config.architectures = ["Qwen3ForCausalLM", architecture]
    assert not supports_non_contiguous_kv_cache(config)


@pytest.mark.parametrize("architecture", ["Qwen3ForCausalLM", "Qwen3_5ForConditionalGeneration"])
@pytest.mark.parametrize(
    "layer_types", [[], ["linear_attention"], ["full_attention", "sparse_attention"], "full_attention"]
)
def test_incompatible_attention_patterns_are_not_enabled(architecture, layer_types):
    assert not supports_non_contiguous_kv_cache(_config(architecture, layer_types))


@pytest.mark.parametrize("option", ["use_sliding_window", "index_topk", "sparse_attention_config"])
def test_sparse_and_windowed_configs_are_not_enabled(option):
    assert not supports_non_contiguous_kv_cache(_config("Qwen3ForCausalLM", **{option: True}))


def test_mla_and_missing_model_metadata_are_not_enabled():
    config = _config("Qwen3ForCausalLM")
    config.model_config.use_mla = True
    assert not supports_non_contiguous_kv_cache(config)
    del config.model_config.architecture
    assert not supports_non_contiguous_kv_cache(config)
    config.model_config.architecture = "Qwen3ForCausalLM"
    config.model_config.hf_text_config = None
    assert not supports_non_contiguous_kv_cache(config)


def test_hybrid_requires_a_known_attention_pattern():
    assert not supports_non_contiguous_kv_cache(_config("Qwen3_5ForConditionalGeneration"))
    assert not supports_non_contiguous_kv_cache(_config("Qwen3_5ForConditionalGeneration", ["linear_attention"] * 4))


def test_dense_architecture_cannot_opt_in_a_recurrent_layer():
    assert not supports_non_contiguous_kv_cache(_config("Qwen3ForCausalLM", ["full_attention", "linear_attention"]))


@pytest.mark.parametrize("architecture", ["Qwen3_5MTP", "Qwen3_5MoeMTP"])
def test_mtp_keeps_full_attention_layout_with_target_text_config(architecture):
    assert supports_non_contiguous_kv_cache(_config(architecture, ["full_attention", "linear_attention"]))


def test_resolved_architecture_is_independent_of_hf_alternatives():
    config = _config("Qwen3ForCausalLM")
    config.model_config.hf_config.architectures = ["UnknownForCausalLM"]
    assert supports_non_contiguous_kv_cache(config)


@pytest.mark.parametrize("runner_version", ["v1", "v2"])
def test_contiguous_hybrid_shared_backing_keeps_group_block_writes_isolated(monkeypatch, runner_version):
    """Legacy K and SSM rows must overlay by block ID, not merely be contiguous."""
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2 import attn_utils

    num_blocks, block_size, conv_elements = 3, 128, 2
    page_bytes = 2 * (2 * block_size + conv_elements)
    attention_spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16, page_size_padded=page_bytes
    )
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((conv_elements,), (block_size,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=page_bytes,
    )
    specs = {"full_attn": attention_spec, "linear_attn": mamba_spec}
    backing_bytes = num_blocks * page_bytes
    cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_bytes, layers=[name], layer_stride=backing_bytes, block_stride=page_bytes, offset=0
            )
            for name in specs
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()],
    )
    config = _config("UnlistedHybridForCausalLM", ["linear_attention", "full_attention"])
    config.additional_config = {}
    config.kv_transfer_config = None
    config.quant_config = None
    config.cache_config = SimpleNamespace(
        cache_dtype="auto",
        get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_layer_compact=True, is_block_compact=True),
    )
    groups = [
        SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=spec, layer_names=[name], backend=AscendAttentionBackend)
        for i, (name, spec) in enumerate(specs.items())
    ]
    if runner_version == "v1":
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.model_config = config.model_config
        runner.vllm_config = config
        runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
        runner.use_sparse = runner.use_compress = runner.sparse_kv_offload_enabled = False
        runner.use_hybrid_blocks = True
        runner.runner_only_attn_layers = set()
        runner.dcp_size = 1
        runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)
        raw = runner._allocate_kv_cache_tensors(cache_config)
        caches = runner._reshape_kv_cache_tensors(cache_config, raw, [64, 1])
    else:
        monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
        raw = attn_utils._allocate_kv_cache(cache_config, shared_layers={}, device=torch.device("cpu"))
        caches = attn_utils._reshape_kv_cache_v2(groups, raw, "auto", [64, 1], {}, cache_config)
    assert raw["full_attn"].untyped_storage().data_ptr() == raw["linear_attn"].untyped_storage().data_ptr()
    key, value = caches["full_attn"]
    conv, ssm = caches["linear_attn"]
    assert all(tensor.is_contiguous() for tensor in (key, value, conv, ssm))
    assert key.data_ptr() == ssm.data_ptr()
    assert key.numel() == ssm.numel()
    assert key.storage_offset() == num_blocks * conv_elements
    # Attention block 0 and Mamba block 1 belong to different live requests.
    key[:2].fill_(3)
    ssm[1].fill_(5)
    conv[1].fill_(7)
    assert torch.all(key[:2] == 3)
    assert torch.all(ssm[0] == 3)
    assert torch.count_nonzero(ssm[2]) == 0
    assert torch.count_nonzero(conv[0]) == torch.count_nonzero(conv[2]) == 0
    assert torch.count_nonzero(value) == 0
