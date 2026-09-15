from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_get_spec_decode_method_dispatches_custom_class():
    vllm_config = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.create_custom_proposer",
        return_value="custom-proposer",
    ) as create:
        proposer = get_spec_decode_method(
            "custom_class",
            vllm_config,
            device="npu",
            runner=MagicMock(),
        )

    assert proposer == "custom-proposer"
    create.assert_called_once_with(vllm_config)


@pytest.mark.parametrize("has_attention_metadata", [True, False])
def test_propose_draft_token_ids_calls_custom_proposer(has_attention_metadata):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(method="custom_class")
    runner.drafter = MagicMock()
    runner.drafter.propose.return_value = [[11, 12], [21, 22]]
    runner.input_batch = SimpleNamespace(
        num_tokens_no_spec=np.array([3, 2], dtype=np.int32),
        token_ids_cpu=np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int32),
    )
    runner._log_propose_draft_token_ids_entry = MagicMock()

    sampled_token_ids = [[3], [5]]
    scheduler_output = SimpleNamespace(num_spec_tokens_to_schedule=2)
    slot_mapping = MagicMock()
    attention_metadata = SimpleNamespace(slot_mapping=slot_mapping) if has_attention_metadata else None
    draft_token_ids = runner.propose_draft_token_ids(
        valid_sampled_token_ids=sampled_token_ids,
        sampling_metadata=MagicMock(),
        scheduler_output=scheduler_output,
        spec_decode_metadata=MagicMock(),
        spec_decode_common_attn_metadata=attention_metadata,
        positions=MagicMock(),
        num_scheduled_tokens=2,
        hidden_states=MagicMock(),
    )

    assert draft_token_ids == [[11, 12], [21, 22]]
    runner.drafter.propose.assert_called_once_with(
        sampled_token_ids,
        runner.input_batch.num_tokens_no_spec,
        runner.input_batch.token_ids_cpu,
        slot_mappings=slot_mapping if has_attention_metadata else None,
    )
