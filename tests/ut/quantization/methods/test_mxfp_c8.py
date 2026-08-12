from unittest.mock import patch

import torch

from tests.ut.base import TestBase
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
