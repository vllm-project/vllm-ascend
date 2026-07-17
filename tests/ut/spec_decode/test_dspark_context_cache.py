from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from vllm_ascend.distributed.kv_transfer.utils.utils import get_dspark_num_kv_cache_layers
from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkAttention
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer


def _make_dspark_attention(kv_cache: torch.Tensor, *, head_dim: int, window_size: int):
    attention = DeepseekV4DSparkAttention.__new__(DeepseekV4DSparkAttention)
    nn.Module.__init__(attention)
    attention.head_dim = head_dim
    attention.window_size = window_size
    attention.scale = 1.0
    attention.attn_sink = nn.Parameter(torch.zeros(1), requires_grad=False)
    attention.dsa_attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=[[kv_cache]])
    )
    attention._dspark_window_offsets = torch.arange(window_size).view(1, -1)
    return attention


def test_dspark_context_kv_uses_paged_slot_mapping() -> None:
    kv_cache = torch.zeros((3, 4, 1, 2), dtype=torch.float32)
    attention = _make_dspark_attention(kv_cache, head_dim=2, window_size=4)
    attention._project_shared_kv = MethodType(lambda _self, hidden, _positions: hidden, attention)

    hidden_states = torch.tensor([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])
    positions = torch.tensor([0, 1, 4], dtype=torch.int32)
    # Logical block 0 is physical block 2; logical block 1 is physical block 0.
    slot_mapping = torch.tensor([8, 9, 0], dtype=torch.int32)

    attention.precompute_context_kv(hidden_states, positions, slot_mapping)

    torch.testing.assert_close(kv_cache[2, 0, 0], hidden_states[0])
    torch.testing.assert_close(kv_cache[2, 1, 0], hidden_states[1])
    torch.testing.assert_close(kv_cache[0, 0, 0], hidden_states[2])


def test_dspark_context_kv_is_read_through_block_table() -> None:
    kv_cache = torch.zeros((3, 4, 1, 2), dtype=torch.float32)
    kv_cache[2, :, 0] = torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    kv_cache[0, 0, 0] = torch.tensor([4.0, 40.0])
    attention = _make_dspark_attention(kv_cache, head_dim=2, window_size=4)
    captured: dict[str, torch.Tensor] = {}

    def capture_attention(_self, q, kv, key_valid):
        captured["kv"] = kv
        captured["key_valid"] = key_valid
        return torch.zeros_like(q)

    attention._dspark_attn = MethodType(capture_attention, attention)
    q = torch.zeros((1, 2, 1, 2))
    draft_kv = torch.tensor([[[5.0, 50.0], [6.0, 60.0]]])
    draft_positions = torch.tensor([[5, 6]], dtype=torch.int32)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)

    attention._dspark_attention_from_cache(q, draft_kv, draft_positions, block_table)

    expected_context = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]])
    torch.testing.assert_close(captured["kv"][:, :4], expected_context)
    assert captured["key_valid"][:, :4].all()


def test_dspark_proposer_selects_its_swa_cache_group() -> None:
    proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
    proposer._draft_attn_layer_names = {
        "model.layers.61.self_attn.swa_cache",
        "model.layers.62.self_attn.swa_cache",
    }
    proposer.model = SimpleNamespace(model=SimpleNamespace(num_dspark_layers=2))
    proposer.num_speculative_tokens = 5
    proposer.max_graph_batch_size = 4
    proposer.max_model_len = 1024
    proposer.device = torch.device("cpu")
    cache_spec = SimpleNamespace(block_size=128)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn.swa_cache"], kv_cache_spec=cache_spec),
            SimpleNamespace(layer_names=sorted(proposer._draft_attn_layer_names), kv_cache_spec=cache_spec),
        ]
    )

    proposer.initialize_attn_backend(kv_cache_config)

    assert proposer.kv_cache_gid == 1
    assert proposer.kernel_block_size == 128
    assert proposer.attn_layer_names == sorted(proposer._draft_attn_layer_names)


def test_dspark_transfers_every_draft_layer() -> None:
    draft_hf_config = SimpleNamespace(
        dspark_block_size=5,
        dspark_num_mtp_layers=3,
        num_hidden_layers=61,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        )
    )

    assert get_dspark_num_kv_cache_layers(vllm_config) == 3


def test_regular_mtp_keeps_existing_transfer_behavior() -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=1)
            ),
        )
    )

    assert get_dspark_num_kv_cache_layers(vllm_config) is None
