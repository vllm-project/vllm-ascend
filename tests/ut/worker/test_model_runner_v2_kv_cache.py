# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.patch.worker.patch_v2 import patch_model_runner  # noqa: F401
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
