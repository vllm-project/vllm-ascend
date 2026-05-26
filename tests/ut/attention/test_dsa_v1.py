from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder


class TestAscendDSAMetadataBuilder(TestBase):
    def test_build_for_graph_capture_supplies_default_cache_dicts(self):
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=16,
                max_num_seqs=8,
            ),
            model_config=SimpleNamespace(
                max_model_len=4096,
                hf_text_config=SimpleNamespace(qk_rope_head_dim=64),
                hf_config=SimpleNamespace(model_type="unit_test_model"),
            ),
            speculative_config=None,
        )
        kv_cache_spec = SimpleNamespace(compress_ratio=4)
        builder = AscendDSAMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["layer_0"],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
        builder.build = MagicMock(return_value=SimpleNamespace(attn_state=None))

        common_attn_metadata = MagicMock()
        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        self.assertEqual(attn_metadata.attn_state, AscendAttentionState.DecodeOnly)
        builder.build.assert_called_once()
        _, kwargs = builder.build.call_args
        self.assertEqual(kwargs["common_prefix_len"], 0)
        self.assertIs(kwargs["common_attn_metadata"], common_attn_metadata)
        self.assertEqual(kwargs["prefill_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["decode_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["common_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["block_size"], 128)