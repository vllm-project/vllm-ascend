import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def test_prefill_seq_lens_update_skips_gdn_metadata_in_hybrid_model():
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "GQA"
    attention_metadata = AscendMetadata(seq_lens=torch.tensor([3, 5]))
    gdn_metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
    )

    speculator._ascend_update_seq_lens(
        {
            "model.layers.0.self_attn": attention_metadata,
            "model.layers.1.linear_attn": gdn_metadata,
        }
    )

    assert torch.equal(attention_metadata.seq_lens, torch.tensor([4, 6]))
    assert attention_metadata.seq_len_list == [4, 6]
    assert not hasattr(gdn_metadata, "seq_lens")
