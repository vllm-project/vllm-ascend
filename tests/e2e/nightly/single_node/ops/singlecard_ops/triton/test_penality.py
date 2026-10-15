import gc

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.penalties import (
    VOCAB_BLOCK_SIZE,
    _penalties_kernel,
    apply_penalties,
)

NUM_TOKENS = [1, 4]
VOCAB_SIZE = [1000]
NUM_STATUS = [1, 4]
NUM_SPECULATIVE_TOKENS = [0, 1, 3]
DTYPES = [torch.bfloat16, torch.float16]
SEEDS = [42]
DEVICES = [f"npu:{0}"]

DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3

# Vocabulary sizes with more than one vocab block AND a partial tail block,
# both asserted below rather than left implicit.
MULTI_BLOCK_VOCAB = [12800, 151936]

# (repetition, frequency, presence). The kernel branches on each of these
# independently, so they need explicit coverage rather than random values.
PENALTY_CONFIGS = {
    "none": (1.0, 0.0, 0.0),
    "rep_only": (1.1, 0.0, 0.0),
    "freq_only": (1.0, 0.1, 0.0),
    "pres_only": (1.0, 0.0, 0.1),
    "rep_pres": (1.1, 0.0, 0.1),
    "all": (1.1, 0.1, 0.1),
}


def pack_prompt_bin_mask(
    prompt_token_ids: list[np.ndarray],
    num_status: int,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    """
    Bit-pack per-request prompt token sets into int32 words.
    """
    num_packed = (vocab_size + 31) // 32
    dense = np.zeros((num_status, num_packed * 32), dtype=np.uint8)
    for row, ids in enumerate(prompt_token_ids):
        dense[row, ids] = 1
    packed = np.packbits(dense, axis=-1, bitorder="little")
    packed = np.ascontiguousarray(packed).view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(packed)).to(device)


def unpack_prompt_bin_mask(prompt_bin_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Inverse of pack_prompt_bin_mask: [R, P] int32 -> [R, vocab_size] bool."""
    arr = np.ascontiguousarray(prompt_bin_mask.detach().cpu().numpy())
    rows = arr.shape[0]
    bits = np.unpackbits(arr.view(np.uint8).reshape(rows, -1), axis=-1, bitorder="little")
    bits = np.ascontiguousarray(bits[:, :vocab_size].astype(bool))
    return torch.from_numpy(bits).to(prompt_bin_mask.device)


def pytorch_apply_penalties(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> torch.Tensor:
    """
    Pytorch equivalent implementation
    """
    num_tokens, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    logits_float = logits.float()

    prompt_masks_unpacked = unpack_prompt_bin_mask(prompt_bin_mask, vocab_size)

    start_idx_in_batch = torch.arange(num_tokens, device=device) - expanded_local_pos

    for token_idx in range(num_tokens):
        req_state_idx = idx_mapping[token_idx].item()

        rep_penalty = repetition_penalty[req_state_idx].item()
        freq_penalty = frequency_penalty[req_state_idx].item()
        pres_penalty = presence_penalty[req_state_idx].item()

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty

        if not use_penalty:
            continue

        current_prompt_mask = prompt_masks_unpacked[req_state_idx]
        base_counts = output_bin_counts[req_state_idx].clone()

        # Compute cumulative draft counts
        pos = expanded_local_pos[token_idx].item()
        draft_counts = torch.zeros(vocab_size, device=device, dtype=torch.int32)

        for prev_pos in range(pos):
            prev_token_idx = start_idx_in_batch[token_idx] + prev_pos + 1
            if 0 <= prev_token_idx < num_tokens:
                prev_token = token_ids[prev_token_idx].item()
                if 0 <= prev_token < vocab_size:
                    draft_counts[prev_token] += 1

        # Total counts = base output counts + cumulative draft counts
        total_counts = base_counts + draft_counts
        output_bin_mask = total_counts > 0

        if use_rep_penalty:
            need_scale = current_prompt_mask | output_bin_mask
            scale = torch.where(need_scale, rep_penalty, 1.0)

            pos_mask = logits_float[token_idx] > 0
            scale_factor = torch.where(pos_mask, 1.0 / scale, scale)
            logits_float[token_idx] *= scale_factor

        if use_freq_penalty:
            logits_float[token_idx] -= freq_penalty * total_counts.float()

        if use_pres_penalty:
            logits_float[token_idx] -= pres_penalty * output_bin_mask.float()

    return logits_float.to(dtype)


def create_test_data(
    num_tokens: int = 8,
    vocab_size: int = 51200,
    num_status: int = 16,
    num_speculative_tokens: int = 3,
    device: str = "npu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    penalty_config: tuple[float, float, float] | None = None,
    spec_decode_groups: bool = False,
):
    """Create test data for penalties.

    penalty_config: if given, every request gets exactly this
        ``(repetition, frequency, presence)`` triple instead of random values.
        The kernel branches on each penalty independently, so pinning them is
        what makes per-branch coverage possible.
    spec_decode_groups: if True, lay the rows out the way ``expand_idx_mapping``
        actually does ``num_status`` runs of ``num_speculative_tokens + 1``
        rows, with ``expanded_local_pos`` ramping 0..k and a constant
        ``req_state_idx`` per run. The default (random per-token values) does
        not correspond to any real batch, since ``start_idx = token_idx - pos``
        then points at arbitrary rows.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    logits = torch.randn(num_tokens, vocab_size, device=device, dtype=dtype)

    if penalty_config is None:
        repetition_penalty = torch.ones(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.3:
                repetition_penalty[i] = torch.rand(1, device=device).item() * 0.8 + 0.6

        frequency_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.5:
                frequency_penalty[i] = torch.rand(1, device=device).item() * 0.2

        presence_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.5:
                presence_penalty[i] = torch.rand(1, device=device).item() * 0.2
    else:
        rep, freq, presence = penalty_config
        repetition_penalty = torch.full((num_status,), rep, device=device, dtype=torch.float32)
        frequency_penalty = torch.full((num_status,), freq, device=device, dtype=torch.float32)
        presence_penalty = torch.full((num_status,), presence, device=device, dtype=torch.float32)

    # Create token_ids for speculative decoding
    token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int32)

    if spec_decode_groups:
        rows_per_req = num_speculative_tokens + 1
        flat_req = np.minimum(np.arange(num_tokens) // rows_per_req, num_status - 1)
        flat_pos = np.arange(num_tokens) % rows_per_req
        idx_mapping = torch.from_numpy(flat_req.astype(np.int32)).to(device)
        expanded_local_pos = torch.from_numpy(flat_pos.astype(np.int32)).to(device)
    else:
        idx_mapping = torch.randint(0, num_status, (num_tokens,), device=device, dtype=torch.int32)
        # Create expanded_local_pos (position within speculative decoding window)
        expanded_local_pos = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        for i in range(num_tokens):
            expanded_local_pos[i] = torch.randint(0, num_speculative_tokens + 1, (1,)).item()

    num_in_prompt = max(1, vocab_size // 20)
    prompt_token_ids = [
        rng.choice(vocab_size, size=num_in_prompt, replace=False).astype(np.int64) for _ in range(num_status)
    ]
    prompt_bin_mask = pack_prompt_bin_mask(prompt_token_ids, num_status, vocab_size, device)

    output_bin_counts = torch.zeros(num_status, vocab_size, device=device, dtype=torch.int32)
    num_output_tokens = max(1, vocab_size // 20)
    out_ids = torch.from_numpy(rng.integers(0, vocab_size, size=(num_status, num_output_tokens), dtype=np.int64)).to(
        device
    )
    out_vals = torch.from_numpy(rng.integers(1, 10, size=(num_status, num_output_tokens), dtype=np.int32)).to(device)
    output_bin_counts.scatter_(1, out_ids, out_vals)

    return (
        logits,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )


def launch_with_num_workers(data, num_workers: int, block_size: int = VOCAB_BLOCK_SIZE) -> None:
    """Run _penalties_kernel with an explicit worker count, in place.

    apply_penalties uses get_vectorcore_num(); this bypasses that so the
    worker-invariance property can be pinned directly.
    """
    (
        logits,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    ) = data
    num_tokens, vocab_size = logits.shape
    num_vocab_blocks = (vocab_size + block_size - 1) // block_size
    total_tiles = num_tokens * num_vocab_blocks
    tiles_per_worker = (total_tiles + num_workers - 1) // num_workers
    _penalties_kernel[(num_workers,)](
        logits,
        logits.stride(0),
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        total_tiles,
        tiles_per_worker,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        BLOCK_SIZE=block_size,
    )


@pytest.fixture(autouse=True)
def _init_triton_device_properties():
    """apply_penalties reads get_vectorcore_num(), which asserts unless device
    properties were initialised. In production that happens once in
    AscendWorker.init_device; tests do not go through the worker, so they
    initialise it themselves, same as every other triton test here."""
    init_device_properties_triton()


def cleanup() -> None:
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


class TestApplyPenalties:
    # make sure the kernel produces the same result as the pytorch implementation
    @pytest.mark.parametrize("num_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
    @pytest.mark.parametrize("num_status", NUM_STATUS)
    @pytest.mark.parametrize("num_speculative_tokens", NUM_SPECULATIVE_TOKENS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_apply_penalties(self, num_tokens, vocab_size, num_status, num_speculative_tokens, dtype, seed, device):
        data = create_test_data(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            num_status=num_status,
            num_speculative_tokens=num_speculative_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        logits_triton = data[0]
        logits_pytorch = logits_triton.clone()

        apply_penalties(*data)

        logits_pytorch_result = pytorch_apply_penalties(logits_pytorch, *data[1:])

        atol = DEFAULT_ATOL
        rtol = DEFAULT_RTOL
        if dtype == torch.bfloat16:
            atol = 1e-02
            rtol = 1e-02
        assert torch.allclose(logits_triton, logits_pytorch_result, atol=atol, rtol=rtol)
        cleanup()

    # The parametrised test above runs at vocab_size=1000, i.e. a single vocab
    # block, so it never exercises the grid-stride loop, the per-block address
    # arithmetic, or the masked tail. These shapes do.
    @pytest.mark.parametrize("vocab_size", MULTI_BLOCK_VOCAB)
    @pytest.mark.parametrize("num_tokens", [1, 64])
    @pytest.mark.parametrize("num_speculative_tokens", [0, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_apply_penalties_multi_vocab_block(self, vocab_size, num_tokens, num_speculative_tokens, dtype, device):
        num_vocab_blocks = (vocab_size + VOCAB_BLOCK_SIZE - 1) // VOCAB_BLOCK_SIZE
        assert num_vocab_blocks > 1, "this test is meant to cover more than one vocab block"
        assert vocab_size % VOCAB_BLOCK_SIZE != 0, "this test is meant to cover the partial tail block"
        num_status = max(1, num_tokens // (num_speculative_tokens + 1))
        data = create_test_data(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            num_status=num_status,
            num_speculative_tokens=num_speculative_tokens,
            device=device,
            dtype=dtype,
            seed=SEEDS[0],
            spec_decode_groups=True,
        )
        logits_triton = data[0]
        logits_pytorch = logits_triton.clone()

        apply_penalties(*data)
        expected = pytorch_apply_penalties(logits_pytorch, *data[1:])

        if dtype == torch.float32:
            # fp32 end to end, so the comparison can be tight enough to be
            # meaningful (bf16 rounding otherwise swamps any real difference).
            atol = rtol = 1e-5
        else:
            atol = rtol = 1e-2
        torch.testing.assert_close(logits_triton, expected, atol=atol, rtol=rtol)
        cleanup()

    # The worker count only controls how the (token, vocab_block) tiles are
    # split across programs; each tile is read-modify-written by exactly one
    # worker, so the result must not depend on it. Pin that, because
    # apply_penalties takes the count from get_vectorcore_num(), which varies
    # by SKU. Counts coprime with num_vocab_blocks force workers to start and
    # end mid-row, which is the only non-trivial indexing in the kernel.
    @pytest.mark.parametrize("vocab_size", MULTI_BLOCK_VOCAB)
    @pytest.mark.parametrize("num_tokens", [1, 5, 64])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_apply_penalties_worker_invariance(self, vocab_size, num_tokens, device):
        n = (vocab_size + VOCAB_BLOCK_SIZE - 1) // VOCAB_BLOCK_SIZE
        assert n > 1, "need more than one vocab block for this to mean anything"

        # 7 and 40 are coprime with n = 38 for Qwen3; 100 exceeds the tile count
        # at num_tokens = 1, leaving workers with nothing to do.
        candidates = sorted({1, 2, 3, 7, n, 40, 48, 100})
        assert len(candidates) > 1, "need at least two worker counts to compare"

        data = create_test_data(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            num_status=num_tokens,
            num_speculative_tokens=0,
            device=device,
            dtype=torch.float32,
            seed=SEEDS[0],
            penalty_config=PENALTY_CONFIGS["all"],
            spec_decode_groups=True,
        )
        logits = data[0]
        pristine = logits.clone()

        reference = None
        for num_workers in candidates:
            logits.copy_(pristine)
            launch_with_num_workers(data, num_workers)
            torch.npu.synchronize()
            if reference is None:
                reference = logits.clone()
            else:
                assert torch.equal(logits, reference), (
                    f"num_workers={num_workers} differs from num_workers={candidates[0]} "
                    f"(max abs diff {(logits - reference).abs().max().item():.3e})"
                )
        cleanup()

    # Each penalty is a separate branch in the kernel; "none" must leave logits
    # untouched via the early return.
    @pytest.mark.parametrize("config_name", sorted(PENALTY_CONFIGS))
    @pytest.mark.parametrize("vocab_size", [MULTI_BLOCK_VOCAB[0]])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_apply_penalties_penalty_configs(self, config_name, vocab_size, device):
        num_tokens = 32
        data = create_test_data(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            num_status=num_tokens,
            num_speculative_tokens=0,
            device=device,
            dtype=torch.float32,
            seed=SEEDS[0],
            penalty_config=PENALTY_CONFIGS[config_name],
            spec_decode_groups=True,
        )
        logits_triton = data[0]
        logits_pytorch = logits_triton.clone()

        apply_penalties(*data)
        expected = pytorch_apply_penalties(logits_pytorch, *data[1:])
        torch.testing.assert_close(logits_triton, expected, atol=1e-5, rtol=1e-5)

        if config_name == "none":
            assert torch.equal(logits_triton, logits_pytorch), "no penalty active, logits must be untouched"
        cleanup()

    # make sure the kernel can handle large shapes without crashing
    @pytest.mark.parametrize(
        "num_tokens,vocab_size,num_status,num_speculative_tokens,dtype",
        [
            pytest.param(
                2048,
                155648,
                4,
                3,
                torch.bfloat16,
                id="target_2048x155648",
            ),
        ],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_apply_penalties_capable_with_large_shape(
        self,
        num_tokens,
        vocab_size,
        num_status,
        num_speculative_tokens,
        dtype,
        seed,
        device,
    ):
        data = create_test_data(
            num_tokens=num_tokens,
            vocab_size=vocab_size,
            num_status=num_status,
            num_speculative_tokens=num_speculative_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )

        apply_penalties(*data)

        # Make asynchronous kernel launch errors fail this test directly.
        torch.npu.synchronize()

        del data
        cleanup()
