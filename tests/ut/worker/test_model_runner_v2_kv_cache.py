# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.attention.attention_v1 import AscendKimiK31GQADraftAttentionImpl
from vllm_ascend.core.kv_cache_interface import AscendGQADraftSlidingWindowSpec
from vllm_ascend.patch.worker.patch_v2 import patch_model_runner  # noqa: F401


@pytest.mark.parametrize("is_profiling", [False, True])
def test_initialize_preserves_connector_containers_and_flattens_runner_cache(is_profiling):
    k, v, conv, ssm, single = [torch.empty(3, 4) for _ in range(5)]
    other = torch.empty(3, 4, device="meta")
    caches = {"attention": (k, None, v), "mamba": [conv, ssm], "single": single, "other": other}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    registration_order = []
    runner._register_sparse_kv_caches.side_effect = lambda _: registration_order.append("offload")

    def register_connector(*_args):
        registration_order.append("connector")

    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value=caches),
        patch.object(upstream, "get_kv_connector", side_effect=register_connector) as connector,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=is_profiling)

    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
    runner._register_sparse_kv_caches.assert_called_once_with(caches)
    if is_profiling:
        assert registration_order == ["offload"]
        connector.assert_not_called()
        assert runner.kv_connector is upstream.NO_OP_KV_CONNECTOR
    else:
        assert registration_order == ["offload", "connector"]
        assert connector.call_args.args[1] is caches
        assert type(caches["attention"]) is tuple
        assert type(caches["mamba"]) is list
        assert caches["attention"][2] is v
        assert caches["mamba"][1] is ssm
        assert caches["other"] is other


@pytest.mark.parametrize(
    ("interleave", "target_supports_interleave", "local_draft_cache", "draft_arch", "error_impl"),
    [
        (768, True, True, "Qwen3DSparkModel", None),
        (896, True, True, "Qwen3DSparkModel", None),
        (896, False, True, "Qwen3DSparkModel", "SimpleNamespace"),
        (896, True, False, "Qwen3DSparkModel", "AscendKimiK31GQADraftAttentionImpl"),
        (896, True, True, "OtherDraft", "AscendKimiK31GQADraftAttentionImpl"),
    ],
)
def test_initialize_checks_attention_cp_using_draft_cache_topology(
    interleave, target_supports_interleave, local_draft_cache, draft_arch, error_impl
):
    runner = MagicMock()
    runner.is_encoder_decoder = False
    runner.device = torch.device("cpu")
    runner.max_model_len = 8192
    runner.vocab_size = 163840
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.dcp_size = 8
    runner.dcp_rank = 1
    runner.parallel_config = SimpleNamespace(
        tensor_parallel_size=8,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=8,
        cp_kv_cache_interleave_size=interleave,
    )
    runner.speculator = MagicMock(spec=upstream.DraftModelSpeculator)
    runner.speculator.draft_attn_layer_names = {"draft"}
    target_impl = SimpleNamespace(
        supports_mtp_with_cp_non_trivial_interleave_size=target_supports_interleave,
        need_to_return_lse_for_decode=True,
    )
    draft_impl = AscendKimiK31GQADraftAttentionImpl.__new__(AscendKimiK31GQADraftAttentionImpl)
    forward_context = {"target": SimpleNamespace(impl=target_impl), "draft": SimpleNamespace(impl=draft_impl)}
    runner.vllm_config = SimpleNamespace(
        mamba_config=None,
        parallel_config=runner.parallel_config,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k31")),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=SimpleNamespace(architectures=[draft_arch])
        ),
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
    )
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    spec_kwargs = {"num_kv_heads": 2, "head_size": 64, "dtype": torch.bfloat16}
    draft_spec = (
        AscendGQADraftSlidingWindowSpec(block_size=128, sliding_window=2048, **spec_kwargs)
        if local_draft_cache
        else FullAttentionSpec(block_size=128, **spec_kwargs)
    )
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["target"], FullAttentionSpec(block_size=interleave, **spec_kwargs)),
            KVCacheGroupSpec(["draft"], draft_spec),
        ],
    )
    message = f"MTP with cp_kv_cache_interleave_size > 1 is not supported in {error_impl}"
    error = pytest.raises(AssertionError, match=message) if error_impl else nullcontext()
    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables") as block_tables,
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "init_kv_cache", return_value={}),
        patch.object(upstream, "get_kv_connector"),
        error,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config)
        if error_impl is None:
            runner.speculator.set_attn.assert_called_once()
    assert block_tables.call_args.kwargs["cp_interleave"] == interleave
    assert block_tables.call_args.kwargs["slot_mapping_enabled"] == [True, not local_draft_cache]
    assert runner.parallel_config.cp_kv_cache_interleave_size == interleave
    assert runner.vllm_config.compilation_config.static_forward_context is forward_context
    assert set(forward_context) == {"target", "draft"}


def test_mrv2_block_copy_preserves_segmented_mamba_storage():
    storage = torch.arange(36, dtype=torch.float32)
    conv = storage[:12].view(3, 4)
    ssm = storage[12:].view(3, 8)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    upstream.copy_kv_cache_blocks_inplace([conv, ssm, conv, ssm], 3, [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)])
    torch.testing.assert_close(conv[2], before_conv[0])
    torch.testing.assert_close(ssm[2], before_ssm[0])
    torch.testing.assert_close(conv[:2], before_conv[:2])
    torch.testing.assert_close(ssm[:2], before_ssm[:2])
