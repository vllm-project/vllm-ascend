# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount


def torch_bincount(mapping, tokens, prompt_len, prefill_len, prompt_bin_mask, output_bin_counts):
    for req in set(mapping.tolist()):
        prompt_tokens = tokens[req, : prompt_len[req]].unique().long()
        packed = torch.zeros(prompt_bin_mask.shape[1], dtype=torch.int64)
        packed.scatter_add_(0, prompt_tokens // 32, 1 << (prompt_tokens % 32))
        prompt_bin_mask[req] = packed.to(torch.int32)
        output_tokens = tokens[req, prompt_len[req] : prefill_len[req]].long()
        output_bin_counts[req] = torch.bincount(
            output_tokens, minlength=output_bin_counts.shape[1]
        ) * mapping.tolist().count(req)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "prompt,prefill,vocab_size,pattern,request_ids",
    [
        (0, 0, 33, "random", []),
        (0, 0, 33, "random", [2, 0]),
        (1, 3, 1, "repeat", [2, 0]),
        (0, 1025, 97, "random", [2, 0]),
        (1025, 1025, 97, "random", [2, 0]),
        (31, 33, 32, "same_word", [2, 0]),
        (31, 63, 4097, "random", [2, 0]),
        (32, 64, 4097, "repeat", [2, 0, 2]),
        (32, 65, 4097, "random", [2, 0]),
        (1023, 1025, 4097, "same_word", [2, 0]),
        (1024, 1025, 4097, "random", [2, 0]),
        (1025, 2050, 4097, "repeat", [2, 0, 2]),
        (5, 3, 151936, "random", [2, 0]),
        (4096, 4128, 151936, "random", [2, 0]),
        (1025, 1, 4097, "random", [2, 0]),
        (1025, 0, 4097, "random", [2, 0]),
    ],
)
def test_bincount_kernel(dtype, prompt, prefill, vocab_size, pattern, request_ids):
    generator = torch.Generator().manual_seed(42)
    rows, width = 4, max(prompt, prefill, 1)
    tokens = torch.randint(vocab_size, (rows, width + 17), generator=generator, dtype=dtype)[:, :width]
    if pattern == "repeat":
        tokens.fill_(min(31, vocab_size - 1))
    elif pattern == "same_word":
        tokens[:] = torch.arange(width).remainder(min(32, vocab_size))
    mapping = torch.tensor(request_ids, dtype=dtype)
    prompt_len = torch.tensor([prompt, 0, max(prompt - 1, 0), 0], dtype=torch.int32)
    prefill_len = torch.tensor([prefill, 0, max(prefill - 1, 0), 0], dtype=torch.int32)
    words = triton.cdiv(vocab_size, 32)
    # Prefix/suffix and row padding detect stray writes; inactive rows retain sentinels.
    mask_storage = torch.full((rows, words + 21), -12345, dtype=torch.int32)
    counts_storage = torch.full((rows, vocab_size + 21), -12345, dtype=torch.int32)
    ref_mask, ref_counts = mask_storage.clone(), counts_storage.clone()
    torch_bincount(
        mapping, tokens, prompt_len, prefill_len, ref_mask[:, 2 : words + 2], ref_counts[:, 2 : vocab_size + 2]
    )

    device_tokens = torch.empty_strided(tokens.shape, tokens.stride(), dtype=dtype, device="npu")
    device_tokens.copy_(tokens)
    device_mapping, device_prompt, device_prefill = [t.npu() for t in (mapping, prompt_len, prefill_len)]
    device_mask, device_counts = mask_storage.npu(), counts_storage.npu()
    mask, counts = device_mask[:, 2 : words + 2], device_counts[:, 2 : vocab_size + 2]
    bincount(device_mapping, device_tokens, device_prompt, device_prefill, mask, counts, int(prefill_len.max()))

    torch.testing.assert_close(device_mask.cpu(), ref_mask, rtol=0, atol=0)
    torch.testing.assert_close(device_counts.cpu(), ref_counts, rtol=0, atol=0)
    for actual, expected in [
        (device_tokens, tokens),
        (device_mapping, mapping),
        (device_prompt, prompt_len),
        (device_prefill, prefill_len),
    ]:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("length", [1047552, 1048576, 1049600])
def test_bincount_launch_limit(length):
    # 64 * ceil(length / 1024) straddles the 65535-program limit in issue #19958.
    requests, rows, vocab_size = 64, 66, 32
    mapping = torch.arange(requests, 0, -1, dtype=torch.int32, device="npu")
    tokens = torch.full((rows, length), 7, dtype=torch.int32, device="npu")
    prompt = torch.full((rows,), length, dtype=torch.int32, device="npu")
    prefill = prompt.clone()
    masks = torch.full((rows, 3), -12345, dtype=torch.int32, device="npu")
    counts = torch.full((rows, vocab_size + 2), -12345, dtype=torch.int32, device="npu")
    bincount(mapping, tokens, prompt, prefill, masks[:, 1:2], counts[:, 1:-1], length)

    expected_mask = torch.full((rows, 3), -12345, dtype=torch.int32)
    expected_counts = torch.full((rows, vocab_size + 2), -12345, dtype=torch.int32)
    expected_mask[1 : requests + 1, 1] = 1 << 7
    expected_counts[1 : requests + 1, 1:-1] = 0
    torch.testing.assert_close(masks.cpu(), expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("rounds,tail", [(1, -1), (1, 0), (1, 1), (2, 1)])
def test_bincount_vector_work_partition(rounds, tail):
    init_device_properties_triton()
    blocks = rounds * get_vectorcore_num() + tail
    length = blocks * 1024
    rows, vocab_size = 4, 4097
    tokens = torch.arange(rows * (length + 17), dtype=torch.int32).reshape(rows, length + 17)
    tokens = tokens.remainder(vocab_size)[:, :length]
    mapping = torch.tensor([2], dtype=torch.int32)
    prompt = torch.tensor([0, 0, length - 1, 0], dtype=torch.int32)
    prefill = torch.tensor([0, 0, length, 0], dtype=torch.int32)
    words = triton.cdiv(vocab_size, 32)
    masks = torch.full((rows, words + 21), -12345, dtype=torch.int32)
    counts = torch.full((rows, vocab_size + 21), -12345, dtype=torch.int32)
    expected_masks, expected_counts = masks.clone(), counts.clone()
    torch_bincount(
        mapping, tokens, prompt, prefill, expected_masks[:, 2 : words + 2], expected_counts[:, 2 : vocab_size + 2]
    )
    token_gpu = torch.empty_strided(tokens.shape, tokens.stride(), dtype=torch.int32, device="npu")
    token_gpu.copy_(tokens)
    mask_gpu, count_gpu = masks.npu(), counts.npu()
    bincount(
        mapping.npu(),
        token_gpu,
        prompt.npu(),
        prefill.npu(),
        mask_gpu[:, 2 : words + 2],
        count_gpu[:, 2 : vocab_size + 2],
        length,
    )
    torch.testing.assert_close(mask_gpu.cpu(), expected_masks, atol=0, rtol=0)
    torch.testing.assert_close(count_gpu.cpu(), expected_counts, atol=0, rtol=0)
    torch.testing.assert_close(token_gpu.cpu(), tokens, atol=0, rtol=0)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
@pytest.mark.parametrize("vocab_size", [5121, 151936])
def test_bincount_then_apply_penalties(vocab_size):
    # Chained entry-point coverage follows PR #15270; include all three policies.
    rows, prompt_tokens, output_tokens = 6, 512, 24
    mapping = torch.tensor([4, 0, 2, 3], dtype=torch.int32)
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(vocab_size, (rows, prompt_tokens + output_tokens), generator=generator, dtype=torch.int32)
    prompt = torch.full((rows,), prompt_tokens, dtype=torch.int32)
    prefill = prompt + output_tokens
    words = triton.cdiv(vocab_size, 32)
    mask_storage = torch.full((rows, words + 21), -12345, dtype=torch.int32)
    count_storage = torch.full((rows, vocab_size + 21), -12345, dtype=torch.int32)
    expected_mask, expected_counts = mask_storage.clone(), count_storage.clone()
    torch_bincount(
        mapping, tokens, prompt, prefill, expected_mask[:, 2 : words + 2], expected_counts[:, 2 : vocab_size + 2]
    )
    device_mapping = mapping.npu()
    device_masks, device_counts = mask_storage.npu(), count_storage.npu()
    masks, counts = device_masks[:, 2 : words + 2], device_counts[:, 2 : vocab_size + 2]
    bincount(device_mapping, tokens.npu(), prompt.npu(), prefill.npu(), masks, counts, int(prefill.max()))

    # Mapped rows exercise frequency-only, presence-only, repetition-only and
    # combined penalties. Unmapped state rows retain their sentinels.
    repetition = torch.tensor([1.0, 1.0, 1.0, 1.5, 1.3, 1.0])
    frequency = torch.tensor([0.4, 0.0, 0.0, 0.3, 0.0, 0.0])
    presence = torch.tensor([0.0, 0.0, 0.2, 0.1, 0.0, 0.0])
    logits = torch.randn(len(mapping), vocab_size, generator=torch.Generator().manual_seed(11))
    expected_logits = logits.clone()
    vocab_ids = torch.arange(vocab_size)
    for token_idx, req in enumerate(mapping.tolist()):
        output_counts = expected_counts[req, 2 : vocab_size + 2].float()
        output_present = output_counts != 0
        packed = expected_mask[req, 2 : words + 2]
        prompt_present = ((packed[vocab_ids // 32] >> (vocab_ids % 32)) & 1) != 0
        scale = torch.where(prompt_present | output_present, repetition[req], 1.0)
        row = expected_logits[token_idx]
        row *= torch.where(row > 0, 1.0 / scale, scale)
        row -= frequency[req] * output_counts
        row -= presence[req] * output_present

    device_logits = logits.npu()
    no_draft_positions = torch.zeros(len(mapping), dtype=torch.int32, device="npu")
    apply_penalties(
        device_logits,
        device_mapping,
        no_draft_positions,
        no_draft_positions,
        repetition.npu(),
        frequency.npu(),
        presence.npu(),
        masks,
        counts,
    )
    torch.testing.assert_close(device_logits.cpu(), expected_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(device_masks.cpu(), expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(device_counts.cpu(), expected_counts, rtol=0, atol=0)
