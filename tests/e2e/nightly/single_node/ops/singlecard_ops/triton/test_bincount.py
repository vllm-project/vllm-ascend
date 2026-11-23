import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.worker.v2.sample.penalties import WORDS_PER_BLOCK, apply_penalties, bincount

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Preserves A2/A3 test scope; A5 nightly validation is outside this PR's verified scope",
)

MAX_NUM_REQS = 64


def torch_bincount(mapping, tokens, prompt_len, prefill_len, prompt_bin_mask, output_bin_counts):
    # Keep the independent oracle on CPU to avoid one NPU synchronization per token.
    request_ids = mapping.cpu().tolist()
    tokens, prompt_len, prefill_len = [tensor.cpu() for tensor in (tokens, prompt_len, prefill_len)]
    masks, counts = prompt_bin_mask.cpu(), output_bin_counts.cpu()
    for req in set(request_ids):
        prompt_ids = tokens[req, : prompt_len[req]].unique().long()
        packed = torch.zeros(masks.shape[1], dtype=torch.int64)
        packed.scatter_add_(0, prompt_ids // 32, 1 << (prompt_ids % 32))
        masks[req] = packed.to(torch.int32)
        counts[req] = torch.bincount(
            tokens[req, prompt_len[req] : prefill_len[req]].long(), minlength=counts.shape[1]
        ) * request_ids.count(req)
    prompt_bin_mask.copy_(masks)
    output_bin_counts.copy_(counts)


def _make_inputs(idx_mapping, prompt_tokens, output_tokens, vocab_size, seed=42):
    """Build kernel inputs; token ids are drawn over the whole vocabulary."""
    torch.manual_seed(seed)
    max_model_len = prompt_tokens + output_tokens + 1
    expanded_idx_mapping = torch.tensor(idx_mapping, dtype=torch.int32).npu()
    all_token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(MAX_NUM_REQS, max_model_len),
        dtype=torch.int32,
    ).npu()
    prompt_len = torch.full((MAX_NUM_REQS,), prompt_tokens, dtype=torch.int32).npu()
    prefill_len = torch.full((MAX_NUM_REQS,), prompt_tokens + output_tokens, dtype=torch.int32).npu()
    num_words = (vocab_size + 31) // 32
    return expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words


@pytest.mark.parametrize(
    "idx_mapping, prompt_tokens, output_tokens",
    [
        # Single request placed at the end of the batch: token_idx 0 must not be
        # confused with req_state_idx 63 anywhere in the kernels.
        ([63], 8, 4),
        # Several requests in a scrambled, non-contiguous order.
        ([63, 7, 31, 2], 2048, 32),
        # Prompt spanning several BLOCK_SIZE=1024 blocks.
        ([17], 4096, 16),
        # No generated tokens yet, so output_bin_counts must stay all zero.
        ([40, 9], 1024, 0),
    ],
)
@pytest.mark.parametrize(
    "vocab_size",
    [
        151936,  # divisible by 32
        151000,  # not divisible by 32: exercises the tail-word masking
        5121,  # small and not divisible by 32
    ],
)
def test_bincount(idx_mapping, prompt_tokens, output_tokens, vocab_size):
    """The packed prompt bitmask and the output token counts must match torch."""
    expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words = _make_inputs(
        idx_mapping, prompt_tokens, output_tokens, vocab_size
    )

    # Poison the outputs so a row or word that is never written is caught
    # instead of silently matching a zero-initialised reference.
    prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()
    ref_prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    ref_output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        prompt_tokens + output_tokens,
    )

    torch_bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        ref_prompt_bin_mask,
        ref_output_bin_counts,
    )

    touched = expanded_idx_mapping.long()
    assert torch.equal(prompt_bin_mask[touched], ref_prompt_bin_mask[touched]), (
        f"prompt_bin_mask differs from torch reference for rows {idx_mapping} at vocab_size={vocab_size}"
    )
    assert torch.equal(output_bin_counts[touched], ref_output_bin_counts[touched]), (
        f"output_bin_counts differs from torch reference for rows {idx_mapping}"
    )
    # Rows outside expanded_idx_mapping must be left untouched.
    untouched = torch.ones(MAX_NUM_REQS, dtype=torch.bool)
    untouched[torch.tensor(idx_mapping)] = False
    untouched = untouched.npu()
    assert torch.all(prompt_bin_mask[untouched] == -1), "bincount wrote outside expanded_idx_mapping"
    assert torch.all(output_bin_counts[untouched] == -1), "bincount wrote outside expanded_idx_mapping"


def torch_apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
):
    vocab_size = logits.shape[1]
    bit_index = torch.arange(vocab_size, device=logits.device)
    for token_idx in range(expanded_idx_mapping.shape[0]):
        req = expanded_idx_mapping[token_idx].item()
        rep = repetition_penalty[req].item()
        freq = frequency_penalty[req].item()
        presence = presence_penalty[req].item()
        if rep == 1.0 and freq == 0.0 and presence == 0.0:
            continue
        counts = output_bin_counts[req].to(torch.float32)
        out_mask = counts != 0
        row = logits[token_idx].to(torch.float32)
        if rep != 1.0:
            words = prompt_bin_mask[req][bit_index // 32]
            prompt_mask = ((words >> (bit_index % 32)) & 1) != 0
            scale = torch.where(prompt_mask | out_mask, rep, 1.0)
            row = row * torch.where(row > 0, 1.0 / scale, scale)
        row = row - freq * counts
        row = row - presence * out_mask.to(torch.float32)
        logits[token_idx] = row


@pytest.mark.parametrize("vocab_size", [5121, 151936])
def test_bincount_then_apply_penalties(vocab_size):
    """bincount feeds apply_penalties: check the chained result, not just bincount."""
    idx_mapping = [63, 7, 31, 2]
    prompt_tokens, output_tokens = 512, 24
    expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words = _make_inputs(
        idx_mapping, prompt_tokens, output_tokens, vocab_size, seed=7
    )

    prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()
    ref_prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    ref_output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        prompt_tokens + output_tokens,
    )
    torch_bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        ref_prompt_bin_mask,
        ref_output_bin_counts,
    )

    num_tokens = len(idx_mapping)
    torch.manual_seed(11)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32).npu()
    ref_logits = logits.clone()
    # No draft tokens, so every logits row is the last position of its request.
    expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int32).npu()
    token_ids = all_token_ids[expanded_idx_mapping.long()]

    repetition_penalty = torch.full((MAX_NUM_REQS,), 1.3, dtype=torch.float32).npu()
    frequency_penalty = torch.full((MAX_NUM_REQS,), 0.4, dtype=torch.float32).npu()
    presence_penalty = torch.full((MAX_NUM_REQS,), 0.2, dtype=torch.float32).npu()

    apply_penalties(
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )
    torch_apply_penalties(
        ref_logits,
        expanded_idx_mapping,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        ref_prompt_bin_mask,
        ref_output_bin_counts,
    )

    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-5)


def _assert_bincount_case(request_ids, prompts, prefills, vocab_size, dtype=torch.int32, tokens=None):
    rows = len(prompts)
    width = max(max(prompts), max(prefills), 1)
    if tokens is None:
        generator = torch.Generator().manual_seed(71)
        tokens = torch.randint(vocab_size, (rows, width + 7), generator=generator, dtype=dtype)[:, :width]
    mapping = torch.tensor(request_ids, dtype=dtype)
    prompt = torch.tensor(prompts, dtype=torch.int32)
    prefill = torch.tensor(prefills, dtype=torch.int32)
    words = (vocab_size + 31) // 32
    masks = torch.full((rows, words + 5), -12345, dtype=torch.int32)
    counts = torch.full((rows, vocab_size + 9), -12345, dtype=torch.int32)
    expected_masks, expected_counts = masks.clone(), counts.clone()
    torch_bincount(mapping, tokens, prompt, prefill, expected_masks[:, 1:-4], expected_counts[:, 2:-7])

    device_tokens = torch.empty_strided(tokens.shape, tokens.stride(), dtype=dtype, device="npu")
    device_tokens.copy_(tokens)
    device_mapping, device_prompt, device_prefill = [tensor.npu() for tensor in (mapping, prompt, prefill)]
    device_masks, device_counts = masks.npu(), counts.npu()
    bincount(
        device_mapping,
        device_tokens,
        device_prompt,
        device_prefill,
        device_masks[:, 1:-4],
        device_counts[:, 2:-7],
        max(prefills),
    )
    torch.testing.assert_close(device_masks.cpu(), expected_masks, rtol=0, atol=0)
    torch.testing.assert_close(device_counts.cpu(), expected_counts, rtol=0, atol=0)
    for actual, expected in [
        (device_tokens, tokens),
        (device_mapping, mapping),
        (device_prompt, prompt),
        (device_prefill, prefill),
    ]:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vocab_size", [1, 33, 4097, 151000])
def test_bincount_ragged_strides(dtype, vocab_size):
    _assert_bincount_case([3, 0, 2], [1025, 0, 31, 2049], [1058, 0, 1024, 2050], vocab_size, dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bincount_duplicate_mapping(dtype):
    _assert_bincount_case([3, 0, 3], [1025, 0, 31, 2049], [1058, 0, 1024, 2050], 4097, dtype)


@pytest.mark.parametrize("mapping", [[], [2, 0]])
def test_bincount_empty_history(mapping):
    _assert_bincount_case(mapping, [0, 0, 0], [0, 0, 0], 33)


def test_bincount_all_flag_byte_patterns():
    # Exercise all 16 combinations of four flag bytes in every nibble position,
    # including bit31, followed by one valid bit in a partially filled word.
    vocab_size = 16 * 32 + 1
    prompt_ids = [word * 32 + bit for word in range(16) for bit in range(32) if word & (1 << (bit % 4))]
    prompt_ids.append(vocab_size - 1)
    tokens = torch.tensor([prompt_ids] * 3, dtype=torch.int32)
    length = len(prompt_ids)
    _assert_bincount_case([2], [0, 0, length], [0, 0, length], vocab_size, tokens=tokens)


@pytest.mark.parametrize("stage", ["statistics", "packing"])
@pytest.mark.parametrize("rounds,tail", [(1, -1), (1, 0), (1, 1), (2, 1)])
def test_bincount_vector_tile_boundaries(stage, rounds, tail):
    init_device_properties_triton()
    blocks = rounds * get_vectorcore_num() + tail
    if stage == "statistics":
        prompt, prefill, vocab_size = blocks * 1024 - 3, blocks * 1024, 4097
    else:
        prompt, prefill = 1025, 1057
        vocab_size = (blocks - 1) * WORDS_PER_BLOCK * 32 + 33
    _assert_bincount_case([2], [0, 0, prompt], [0, 0, prefill], vocab_size)


@pytest.mark.parametrize("length", [1047552, 1048576, 1049600])
def test_bincount_issue_19958(length):
    # Exact issue geometry and repeated token7: 65472/65536/65600 old programs.
    requests, vocab_size = 64, 151936
    mapping = torch.arange(requests, dtype=torch.int32, device="npu")
    tokens = torch.full((requests, length), 7, dtype=torch.int32, device="npu")
    prompt = torch.full((requests,), length, dtype=torch.int32, device="npu")
    masks = torch.full((requests, (vocab_size + 31) // 32), -1, dtype=torch.int32, device="npu")
    counts = torch.full((requests, vocab_size), -1, dtype=torch.int32, device="npu")
    bincount(mapping, tokens, prompt, prompt, masks, counts, length)
    expected_masks = torch.zeros(masks.shape, dtype=torch.int32)
    expected_masks[:, 0] = 1 << 7
    torch.testing.assert_close(masks.cpu(), expected_masks, rtol=0, atol=0)
    assert torch.count_nonzero(counts).item() == 0


@pytest.mark.parametrize("requests", [8, 64])
@pytest.mark.parametrize("prompt,output", [(1, 0), (8, 4), (32, 32)])
def test_bincount_short_request_admission(requests, prompt, output):
    # These are legal text-token histories, including more requests than Vector
    # cores. Preserve their exact results even when main has lower device work.
    rows = requests + 2
    _assert_bincount_case(list(range(requests)), [prompt] * rows, [prompt + output] * rows, 151936)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bincount_reuses_request_rows(dtype):
    rows, width, vocab_size = 3, 3072, 97
    tokens = torch.arange(width, dtype=dtype).remainder(vocab_size).repeat(rows, 1)
    tokens[:, 0] = 31
    device_tokens = tokens.npu()
    words = (vocab_size + 31) // 32
    masks = torch.full((rows, words), -12345, dtype=torch.int32)
    counts = torch.full((rows, vocab_size), -12345, dtype=torch.int32)
    device_masks, device_counts = masks.npu(), counts.npu()
    scenarios = [
        ([2, 0], [2048, 0, 2048], [2050, 0, 2050]),
        # Row2 now has only bit31 and no output history. Row0 retains the
        # previous invocation's result despite containing stale nonzero data.
        ([2], [2048, 0, 1], [2050, 0, 1]),
    ]
    for request_ids, prompts, prefills in scenarios:
        mapping = torch.tensor(request_ids, dtype=dtype)
        prompt = torch.tensor(prompts, dtype=torch.int32)
        prefill = torch.tensor(prefills, dtype=torch.int32)
        torch_bincount(mapping, tokens, prompt, prefill, masks, counts)
        bincount(
            mapping.npu(),
            device_tokens,
            prompt.npu(),
            prefill.npu(),
            device_masks,
            device_counts,
            max(prefills[index] for index in request_ids),
        )
        torch.testing.assert_close(device_masks.cpu(), masks, rtol=0, atol=0)
        torch.testing.assert_close(device_counts.cpu(), counts, rtol=0, atol=0)
        torch.testing.assert_close(device_tokens.cpu(), tokens, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shift", [0, 7])
def test_bincount_cross_program_same_word(dtype, shift):
    init_device_properties_triton()
    blocks = max(32, 2 * get_vectorcore_num() + 1)
    length = blocks * 1024
    storage = torch.empty((3, length + 7), dtype=dtype)
    tokens = storage[:, :length]
    for block in range(blocks):
        # Every block writes a different bit of word0, cycling over all32
        # bits. A request spans enough history tiles to use multiple programs.
        tokens[:, block * 1024 : (block + 1) * 1024] = (13 * block + shift) % 32
    tokens[:, -1] = 32
    _assert_bincount_case([2], [0, 0, length], [0, 0, length], 33, dtype, tokens=tokens)
