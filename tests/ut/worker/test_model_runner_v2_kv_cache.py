# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, HiddenStateCacheSpec, KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.patch.worker.patch_v2.patch_model_runner import init_kv_zero_meta
from vllm_ascend.worker import utils as ascend_utils


@pytest.mark.parametrize("is_profiling", [False, True])
def test_initialize_preserves_connector_containers_and_flattens_runner_cache(is_profiling):
    k, v, conv, ssm, single = [torch.empty(3, 4) for _ in range(5)]
    other = torch.empty(3, 4, device="meta")
    caches = {"attention": (k, v), "mamba": [conv, ssm], "single": single, "other": other}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])

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
        patch.object(upstream, "get_kv_connector") as connector,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=is_profiling)

    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
    if is_profiling:
        connector.assert_not_called()
        assert runner.kv_connector is upstream.NO_OP_KV_CONNECTOR
    else:
        assert connector.call_args.args[1] is caches
        assert type(caches["attention"]) is tuple
        assert type(caches["mamba"]) is list
        assert caches["attention"][1] is v
        assert caches["mamba"][1] is ssm
        assert caches["other"] is other


def test_mrv2_block_copy_preserves_segmented_mamba_storage():
    storage = torch.arange(36, dtype=torch.float32)
    conv = storage[:12].view(3, 4)
    ssm = storage[12:].view(3, 8)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    with patch.object(ascend_utils, "async_tensor_h2d", side_effect=lambda data, **kw: torch.as_tensor(data, **kw)):
        upstream.copy_kv_cache_blocks_inplace(
            [conv, ssm, conv, ssm], 3, [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]
        )
    torch.testing.assert_close(conv[2], before_conv[0])
    torch.testing.assert_close(ssm[2], before_ssm[0])
    torch.testing.assert_close(conv[:2], before_conv[:2])
    torch.testing.assert_close(ssm[:2], before_ssm[:2])


@pytest.mark.parametrize(("block_size", "num_blocks"), [(16, 2), (32, 3)])
def test_mrv2_zeroer_metadata_and_hidden_state_exclusion(block_size, num_blocks):
    ratio = block_size // 16
    k, v = torch.empty(num_blocks * ratio, 4), torch.empty(num_blocks * ratio, 4)
    hidden = torch.full((num_blocks, 1, 16, 4), 7.0)
    attention_spec = FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=4, dtype=torch.float32)
    hidden_spec = HiddenStateCacheSpec(block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float32)
    groups = [
        SimpleNamespace(kv_cache_spec=attention_spec, kv_cache_group_id=0, layer_names=["attention"]),
        SimpleNamespace(kv_cache_spec=hidden_spec, kv_cache_group_id=1, layer_names=["hidden"]),
    ]
    context = {
        "attention": SimpleNamespace(kv_cache=(k, v)),
        "hidden": SimpleNamespace(kv_cache=hidden),
    }
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        attn_groups=[[group] for group in groups],
        kernel_block_sizes=[16, 16],
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    with patch("vllm_ascend.patch.worker.patch_v2.patch_model_runner.is_pin_memory_available", return_value=False):
        init_kv_zero_meta(runner)
    assert type(runner.kv_block_zeroer) is ascend_utils.AscendKVBlockZeroer
    addresses, page_size, _, count = runner.kv_block_zeroer._meta
    assert page_size == 4 * ratio
    assert addresses.tolist() == [k.data_ptr(), v.data_ptr()]
    assert count == 2
    assert context["attention"].kv_cache[0] is k
    assert context["attention"].kv_cache[1] is v
    assert context["hidden"].kv_cache is hidden
    # Two blocks must also be excluded: len(tensor) == 2 is not a K/V tuple.
    torch.testing.assert_close(hidden, torch.full_like(hidden, 7.0))
