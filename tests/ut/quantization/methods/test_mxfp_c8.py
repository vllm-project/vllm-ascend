from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendC8MXFPAttentionBackend,
    AscendC8MXFPAttentionBackendImpl,
)
from vllm_ascend.quantization.methods.mxfp_c8 import (
    AscendC8MXFPKVCacheAttentionMethod,
)


class TestAscendC8MXFPKVCacheAttentionMethod(TestBase):
    def test_missing_v_scale_uses_e8m0_unity_default(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(
            AscendC8MXFPKVCacheAttentionMethod
        )
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4

        with patch.object(
            layer,
            "register_parameter",
            wraps=layer.register_parameter,
        ):
            method.create_weights(layer)

        self.assertEqual(layer.v_cache_scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.v_cache_scale, torch.full((8,), 127, dtype=torch.uint8)))

    def test_installs_c8_backend_with_512_token_blocks(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(
            AscendC8MXFPKVCacheAttentionMethod
        )
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        layer.impl = object.__new__(AscendC8MXFPAttentionBackendImpl.__base__)

        method.create_weights(layer)

        self.assertIs(layer.attn_backend, AscendC8MXFPAttentionBackend)
        self.assertIsInstance(layer.impl, AscendC8MXFPAttentionBackendImpl)
        self.assertEqual(AscendAttentionBackend.get_supported_kernel_block_sizes(), [128])
        self.assertEqual(AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes(), [512])
