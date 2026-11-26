# SPDX-License-Identifier: Apache-2.0
"""NPU parity for direct request-sharded cache reads, including recovery draws."""

import gc

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.resample import resample
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample

if not vllm_version_is("0.33.0"):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

if not (hasattr(torch, "npu") and torch.npu.is_available()):
    pytest.skip("NPU required", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _device():
    init_device_properties_triton()
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


def _case(dtype, vocab):
    # Batch positions, original state slots and cache rows are all different.
    states = [5, 2, 7, 3]
    lengths = [3, 3, 3, 1]
    generator = torch.Generator().manual_seed(20261125)
    q = torch.randn(2, 2, vocab, generator=generator).to(device="npu", dtype=dtype)
    target = torch.randn(sum(lengths), vocab, generator=generator).to(device="npu", dtype=dtype)
    dense = torch.zeros(8, 2, vocab, dtype=dtype, device="npu")
    dense[5] = q[1]
    dense[2] = q[0]
    mapping = torch.tensor([-1, -1, 0, -1, -1, 1, -1, -1], dtype=torch.int32, device="npu")
    temperature = torch.tensor([1, 1, 1.3, 0, 1, 0.7, 1, 0.9], device="npu", dtype=torch.float32)
    seeds = torch.arange(8, device="npu", dtype=torch.int64) * 104729 + 17
    expanded = torch.tensor([s for s, n in zip(states, lengths) for _ in range(n)],
                            device="npu", dtype=torch.int32)
    positions = torch.arange(sum(lengths), device="npu", dtype=torch.int64) * 7 + 19
    return dict(
        q=q, dense=dense, mapping=mapping, temperature=temperature, seed=seeds,
        target=target, expanded=expanded, positions=positions,
        cu=torch.tensor([0, 3, 6, 9, 10], device="npu", dtype=torch.int32),
        idx=torch.tensor(states, device="npu", dtype=torch.int32),
        local_pos=torch.tensor([0, 1, 2] * 3 + [0], device="npu", dtype=torch.int32),
        drafts=torch.tensor([0, 1, 2] * 3 + [0], device="npu", dtype=torch.int64),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [1031, 8193])
@torch.inference_mode()
def test_resample_reads_mapped_rows_and_preserves_request_rng(dtype, vocab):
    c = _case(dtype, vocab)
    rejected = torch.tensor([0, 1, 0, 0], device="npu", dtype=torch.int32)
    starts = [0, 3, 6, 9]
    states = [5, 2, 7, 3]
    target_lse = torch.stack([
        torch.logsumexp(c["target"][start + step].float(), dim=0)
        for start, step in zip(starts, [0, 1, 0, 0])
    ])
    draft_lse = torch.stack([
        torch.logsumexp(c["dense"][state, step].float() / c["temperature"][state].clamp(min=1e-6), dim=0)
        for state, step in zip(states, [0, 1, 0, 0])
    ])
    # State 7 has no local q row. Its mapped call must use target recovery,
    # matching an explicitly invalidated draft in the replicated reference.
    invalid_reference = c["drafts"].clone()
    invalid_reference[7] = -1
    outputs = []
    before = c["q"].clone()
    for q, mapping, draft in [(c["dense"], None, invalid_reference),
                              (c["q"], c["mapping"], c["drafts"])]:
        sampled = torch.full((4, 3), -123, device="npu", dtype=torch.int64)
        counts = rejected.clone()
        resample(
            sampled, counts, c["target"], target_lse, q, draft_lse, c["cu"],
            c["expanded"], draft, c["temperature"], c["seed"], c["positions"],
            has_draft_logits=True, draft_logits_row_mapping=mapping,
        )
        outputs.append((sampled, counts))
    torch.npu.synchronize()
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(c["q"], before, rtol=0, atol=0)


@pytest.mark.skipif(vllm_version_is("0.33.0"), reason="Requires Core DraftLogitsCache and mapped stats ABI")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_verification", [False, True])
@torch.inference_mode()
def test_complete_rejection_preserves_stale_ownership_and_bonus(dtype, block_verification):
    c = _case(dtype, 8193)
    cache = object.__new__(DraftLogitsCache)
    cache.values = c["q"]
    # State 7's mapping is stale: row zero was overwritten by state 2.
    cache.row_mapping = c["mapping"].clone()
    cache.row_mapping[7] = 0
    cache.inverse_row_mapping = torch.tensor([2, 5], device="npu", dtype=torch.int32)
    reference_drafts = c["drafts"].clone()
    reference_drafts[7:9] = -1
    outputs = []
    before = cache.values.clone()
    for q, draft in [(c["dense"], reference_drafts), (cache, c["drafts"])]:
        outputs.append(rejection_sample(
            c["target"], q, draft, c["cu"], c["positions"], c["idx"],
            c["expanded"], c["local_pos"], c["temperature"], c["seed"],
            num_speculative_steps=2, use_block_verification=block_verification,
            use_entropy_verify=False,
        ))
    torch.npu.synchronize()
    ref_tokens, ref_counts = outputs[0]
    actual_tokens, actual_counts = outputs[1]
    torch.testing.assert_close(actual_counts, ref_counts, rtol=0, atol=0)
    valid = torch.arange(3, device="npu")[None, :] < ref_counts[:, None]
    torch.testing.assert_close(actual_tokens[valid], ref_tokens[valid], rtol=0, atol=0)
    torch.testing.assert_close(cache.values, before, rtol=0, atol=0)
