import torch

from vllm_ascend.worker.structured_output import build_invalid_draft_mask


def test_invalid_drafts_start_after_first_padding_token():
    mask = build_invalid_draft_mask(
        req_ids=["grammar", "plain"],
        grammar_req_ids=["grammar"],
        scheduled_spec_decode_tokens={"grammar": [7, 7, -1, -1]},
        num_draft_tokens=[4, 2],
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask.tolist() == [False, False, True, True, False, False]
