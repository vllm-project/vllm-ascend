# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.uno import (
    MAX_ASCEND_UNO_SPECULATIVE_TOKENS,
    UPSTREAM_UNO_AVAILABLE,
    AscendUnoProposer,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

pytestmark = pytest.mark.skipif(
    not UPSTREAM_UNO_AVAILABLE,
    reason="requires a vLLM revision containing Uno",
)


@pytest.fixture
def proposer() -> AscendUnoProposer:
    drafter = object.__new__(AscendUnoProposer)
    drafter.device = torch.device("cpu")
    drafter.num_speculative_tokens = 3
    drafter.speculative_config = SimpleNamespace(
        num_speculative_tokens=3,
        uno_noise_seed=17,
        uno_mask_token_id=31,
    )
    drafter.max_model_len = 32
    drafter.max_batch_size = 2
    drafter.block_size = 4
    drafter.uno_lora_id = 7
    drafter._step = 0
    drafter._lora_hook = None
    drafter._last_draft_probs = None
    drafter._pending_lora_map = ()
    drafter.input_ids = torch.zeros(6, dtype=torch.int32)
    drafter.positions = torch.zeros(6, dtype=torch.int64)
    drafter._slot_mapping_buffer = torch.full((6,), PADDING_SLOT_ID)
    drafter._draft_attn_layer_names = {"attention"}
    drafter.use_local_argmax_reduction = False
    drafter.use_heterogeneous_vocab = False
    drafter._enable_probabilistic_draft_probs = False
    drafter.use_fp64_gumbel = False
    drafter.vllm_config = Mock()
    return drafter


@pytest.fixture
def context() -> AscendCommonAttentionMetadata:
    seq_lens_cpu = torch.tensor([7, 16], dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=seq_lens_cpu.clone(),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu.clone(),
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        max_seq_len=16,
        block_table_tensor=torch.tensor(
            [
                [2, 4, 6, 8, 10, 12, 14, 16],
                [3, 5, 7, 9, 11, 13, 15, 17],
            ],
            dtype=torch.int32,
        ),
        slot_mapping=torch.zeros(8, dtype=torch.int64),
        actual_seq_lengths_q=[4, 8],
        positions=torch.tensor([3, 4, 5, 6, 12, 13, 14, 15]),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )


def _inputs(rejected: torch.Tensor | None = None) -> dict[str, object]:
    return {
        "target_token_ids": torch.zeros(8, dtype=torch.int32),
        "target_positions": torch.tensor([3, 4, 5, 6, 12, 13, 14, 15]),
        "target_hidden_states": torch.zeros(8, 4),
        "next_token_ids": torch.tensor([29, 30], dtype=torch.int32),
        "token_indices_to_sample": None,
        "num_rejected_tokens_gpu": rejected,
    }


def test_factory_dispatches_uno() -> None:
    sentinel = object()
    with patch(
        "vllm_ascend.spec_decode.AscendUnoProposer",
        return_value=sentinel,
    ) as constructor:
        from vllm_ascend.spec_decode import get_spec_decode_method

        config = object()
        device = object()
        runner = object()
        result = get_spec_decode_method("uno", config, device, runner)

    assert result is sentinel
    constructor.assert_called_once_with(config, device, runner)


@pytest.mark.parametrize(
    ("rejected", "positions", "slots", "seq_lens"),
    [
        (None, [7, 8, 9, 16, 17, 18], [19, 24, 25, 44, 45, 46], [10, 19]),
        ([1, 2], [6, 7, 8, 14, 15, 16], [18, 19, 24, 38, 39, 44], [9, 17]),
    ],
)
def test_ascend_metadata_follows_accepted_prefix(
    proposer: AscendUnoProposer,
    context: AscendCommonAttentionMetadata,
    rejected: list[int] | None,
    positions: list[int],
    slots: list[int],
    seq_lens: list[int],
) -> None:
    rejected_tensor = None if rejected is None else torch.tensor(rejected)
    num_tokens, sample_indices, metadata = proposer.set_inputs_first_pass(**_inputs(rejected_tensor), cad=context)

    assert num_tokens == 6
    assert sample_indices.tolist() == list(range(6))
    assert proposer.positions.tolist() == positions
    assert metadata.slot_mapping.tolist() == slots
    assert metadata.seq_lens.tolist() == seq_lens
    assert metadata.query_start_loc.tolist() == [0, 3, 6]
    assert metadata.query_start_loc_cpu.tolist() == [0, 3, 6]
    assert metadata.actual_seq_lengths_q == [3, 6]
    assert metadata.decode_token_per_req == 3
    assert metadata.attn_state is AscendAttentionState.ChunkedPrefill
    if rejected is None:
        assert metadata.seq_lens_cpu.tolist() == seq_lens
    else:
        assert metadata.seq_lens_cpu is None
    assert proposer.input_ids[[0, 3]].tolist() == [29, 30]
    assert proposer._pending_lora_map == (0, 7, 7, 0, 7, 7)
    assert context.seq_lens.tolist() == [7, 16]


def test_propose_uses_ascend_context_and_restores_lora(
    proposer: AscendUnoProposer,
    context: AscendCommonAttentionMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_mappings: list[tuple[int, ...] | None] = []
    proposer.set_lora_hook(active_mappings.append)
    proposer.build_per_group_and_layer_attn_metadata = Mock(return_value=([], {}))
    forward_context = Mock(return_value=nullcontext())
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.uno.set_ascend_forward_context",
        forward_context,
    )

    class Model:
        def __call__(self, input_ids, positions, inputs_embeds):
            assert active_mappings[-1] == (0, 7, 7, 0, 7, 7)
            return torch.nn.functional.one_hot(input_ids.long(), num_classes=32).float()

        def compute_logits(self, hidden_states):
            return hidden_states

    proposer.model = Model()
    drafts = proposer.propose(
        3,
        **_inputs(),
        common_attn_metadata=context,
        sampling_metadata=SimpleNamespace(all_greedy=True),
    )

    assert drafts.tolist() == proposer.input_ids.reshape(2, 3).tolist()
    assert active_mappings == [(0, 7, 7, 0, 7, 7), None]
    assert forward_context.call_args.kwargs["aclgraph_runtime_mode"].name == "NONE"
    assert proposer._pending_lora_map == ()


def test_ascend_uno_rejects_width_beyond_attention_limit() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_uno=lambda: True,
            num_speculative_tokens=MAX_ASCEND_UNO_SPECULATIVE_TOKENS + 1,
        )
    )
    proposer = object.__new__(AscendUnoProposer)
    with pytest.raises(ValueError, match="num_speculative_tokens <= 15"):
        proposer.__init__(config, SimpleNamespace(type="npu"))


def test_runner_uno_lora_reloads_and_restores_base_mapping() -> None:
    runner = object.__new__(NPUModelRunner)
    runner._ensure_lora_enabled = Mock()
    runner._set_active_loras = Mock()
    registered: set[int] = set()
    runner.lora_manager = SimpleNamespace(
        list_adapters=lambda: registered.copy(),
        add_adapter=Mock(side_effect=lambda request: registered.add(request.lora_int_id)),
    )
    request = LoRARequest("uno", 1_000_003, "test-adapter")
    drafter = SimpleNamespace(
        lora_request=request,
        uno_lora_id=request.lora_int_id,
        set_lora_hook=Mock(),
    )

    runner._install_uno_lora(drafter)
    hook = drafter.set_lora_hook.call_args.args[0]
    registered.clear()
    mapping = (0, request.lora_int_id, 0, request.lora_int_id)
    hook(mapping)
    hook(None)

    assert registered == {request.lora_int_id}
    assert runner.lora_manager.add_adapter.call_count == 2
    assert runner._set_active_loras.call_args_list[-2].args == (
        mapping,
        mapping,
        {request},
    )
    assert runner._set_active_loras.call_args_list[-1].args == (
        (0,) * len(mapping),
        (0,) * len(mapping),
        set(),
    )


def test_runner_uno_dispatch_uses_synchronized_cpu_tokens() -> None:
    runner = object.__new__(NPUModelRunner)
    runner._log_propose_draft_token_ids_entry = Mock()
    runner.speculative_config = SimpleNamespace(
        uses_extract_hidden_states=lambda: False,
        use_uno=lambda: True,
    )
    runner.requests = {"request": Mock()}
    runner.input_batch = SimpleNamespace(
        req_ids=["request"],
        num_tokens_no_spec=torch.tensor([1]).numpy(),
    )
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([11], dtype=torch.int32))
    runner._get_positions = Mock(return_value=torch.tensor([0]))
    runner._draft_probs = None
    runner._draft_prob_req_ids = None
    draft_ids = torch.tensor([[21, 22, 23]])
    draft_probs = torch.rand(1, 3, 32)
    drafter = object.__new__(AscendUnoProposer)
    drafter.prepare_next_token_ids_cpu = Mock(return_value=torch.tensor([20], dtype=torch.int32))
    drafter.propose = Mock(return_value=draft_ids)
    drafter.take_last_draft_probs = Mock(return_value=draft_probs)
    runner.drafter = drafter
    scheduler_output = SimpleNamespace(
        num_spec_tokens_to_schedule=3,
        num_scheduled_tokens={"request": 1},
    )
    sampling_metadata = Mock()

    result = runner.propose_draft_token_ids(
        [[20]],
        sampling_metadata,
        scheduler_output,
        None,
        Mock(),
        torch.tensor([0]),
        1,
        torch.zeros(1, 4),
    )

    assert result is draft_ids
    drafter.prepare_next_token_ids_cpu.assert_called_once()
    assert drafter.propose.call_args.kwargs["next_token_ids"].tolist() == [20]
    assert runner._draft_probs is draft_probs
    assert runner._draft_prob_req_ids == ["request"]


def test_runner_rejects_request_lora_before_state_update() -> None:
    runner = object.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(use_uno=lambda: True)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(lora_request=LoRARequest("user", 1, "test-adapter"))]
    )

    with pytest.raises(ValueError, match="request-specific LoRA"):
        runner._update_states(scheduler_output)
