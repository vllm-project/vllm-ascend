import pytest
import torch

from tests.e2e.nightly.single_node.ops.fa3_attention_test_utils import (
    cleanup_npu,
    cosine_similarity_per_req,
    make_fa3_batch,
    make_fa3_decode_case,
    require_a5,
    run_fa3_operator,
)


@pytest.mark.parametrize("seq_lens", [[129, 257], [255, 383]])
@torch.inference_mode()
def test_fa3_batched_decode_matches_single_and_golden(seq_lens: list[int]):
    require_a5()

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    case = make_fa3_decode_case(
        seq_lens=seq_lens,
        num_heads=32,
        latent_dim=512,
        rope_dim=64,
        block_size=128,
        device=device,
        seed=20260421,
    )

    single_outputs = []
    single_golden = []
    for req_id in range(len(seq_lens)):
        single_batch = make_fa3_batch(case, [req_id])
        single_output = run_fa3_operator(single_batch, case)
        single_outputs.append(single_output.squeeze(0))
        single_golden.append(single_batch.golden_output.squeeze(0))

    batched = make_fa3_batch(case, list(range(len(seq_lens))))
    batched_output = run_fa3_operator(batched, case)

    single_outputs_tensor = torch.stack(single_outputs, dim=0)
    single_golden_tensor = torch.stack(single_golden, dim=0)
    batched_vs_single_cos = cosine_similarity_per_req(batched_output, single_outputs_tensor)
    batched_vs_golden_cos = cosine_similarity_per_req(batched_output, batched.golden_output)
    single_vs_golden_cos = cosine_similarity_per_req(single_outputs_tensor, single_golden_tensor)

    torch.testing.assert_close(batched_output.float(), single_outputs_tensor.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(batched_output.float(), batched.golden_output.float(), atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(single_outputs_tensor.float(), single_golden_tensor.float(), atol=6e-2, rtol=6e-2)

    assert torch.all(batched_vs_single_cos > 0.999), batched_vs_single_cos
    assert torch.all(batched_vs_golden_cos > 0.995), batched_vs_golden_cos
    assert torch.all(single_vs_golden_cos > 0.995), single_vs_golden_cos

    cleanup_npu()
