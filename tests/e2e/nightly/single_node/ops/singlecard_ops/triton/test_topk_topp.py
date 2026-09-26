# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch
from torch import Generator
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.v2.sample.topk_topp import apply_top_k_top_p_triton
from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch

# Vocabulary sizes of the models served on NPU (Qwen-class and Llama-class).
VOCAB_SIZES = [1024, 32000, 128256]
BATCH_SIZES = [1, 8, 32, 128, 512]
SEEDS = [42]
DEVICES = [f"npu:{0}"]

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
class TestTopkToppTriton:
    """Compare the Ascend Qrita Triton kernel with the sort+mask reference.

    Reference: `_apply_top_k_top_p_pytorch` (the vllm-ascend v1 sampler
    fallback the kernel replaces). All math is fp32.

    Top-k only is a pure selection: the result must be bit-identical to
    the reference. Top-p involves fp32 probability sums accumulated in
    different orders, so the kept-count boundary may differ slightly:
    rows whose kept count agrees must keep the same positions, and count
    differences are bounded absolutely (<= 3) or relatively (< 0.5%).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.generator = Generator(device=DEVICES[0]).manual_seed(42)

    def _compare_results(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ):
        logits_pytorch = logits.clone()
        logits_triton = logits.clone().to(torch.float32)

        result_pytorch = _apply_top_k_top_p_pytorch(logits_pytorch, k, p)
        result_triton = apply_top_k_top_p_triton(logits_triton, k, p)

        pytorch_mask = result_pytorch != float("-inf")
        triton_mask = result_triton != float("-inf")
        pytorch_kept = pytorch_mask.sum(dim=-1)
        triton_kept = triton_mask.sum(dim=-1)

        if p is None:
            # Top-k only: same selection, values copied verbatim -> bit-exact.
            assert torch.equal(result_pytorch, result_triton), (
                f"Top-k mismatch: PyTorch kept {pytorch_kept.tolist()}, Triton kept {triton_kept.tolist()}"
            )
            return

        # Rows whose kept count agrees must keep the same positions.
        same_count = pytorch_kept == triton_kept
        if same_count.any():
            mismatched = (pytorch_mask != triton_mask).sum(dim=-1)
            assert mismatched[same_count].max().item() == 0, (
                f"Top-p position mismatch on {mismatched[same_count].max().item()} "
                f"token(s): PyTorch kept {pytorch_kept.tolist()}, "
                f"Triton kept {triton_kept.tolist()}"
            )

        # Count tolerance for the fuzzy top-p boundary.
        max_diff = (pytorch_kept - triton_kept).abs().max().item()
        max_kept = pytorch_kept.max().item()
        if max_kept > 0 and max_diff > 3:
            diff_pct = max_diff / max_kept * 100
            assert diff_pct < 0.5, (
                f"Top-p kept-count difference too large: {diff_pct:.2f}% (max diff {max_diff} values out of {max_kept})"
            )

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topk_only(self, vocab_size, batch_size, seed, device):
        """Top-k only (p=None); ~25% of rows disable top-k (k=V)."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        k = torch.randint(1, min(100, vocab_size), (batch_size,), generator=self.generator, device=device).to(
            torch.int32
        )
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator, device=device) == 0
        k.masked_fill_(disable_mask, vocab_size)

        self._compare_results(logits, k, p=None)
        gc.collect()
        torch.npu.empty_cache()

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topp_only(self, vocab_size, batch_size, seed, device):
        """Top-p only (k=None); ~25% of rows disable top-p (p=1.0)."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        p = torch.rand(batch_size, generator=self.generator, device=device) * 0.9 + 0.1
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator, device=device) == 0
        p.masked_fill_(disable_mask, 1.0)

        self._compare_results(logits, k=None, p=p)
        gc.collect()
        torch.npu.empty_cache()

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topk_and_topp(self, vocab_size, batch_size, seed, device):
        """Combined top-k + top-p with random per-row k/p (mixed batch)."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        k = torch.randint(1, min(100, vocab_size), (batch_size,), generator=self.generator, device=device).to(
            torch.int32
        )
        p = torch.rand(batch_size, generator=self.generator, device=device) * 0.9 + 0.1

        disable_k = torch.randint(0, 4, (batch_size,), generator=self.generator, device=device) == 0
        k.masked_fill_(disable_k, vocab_size)
        disable_p = torch.randint(0, 4, (batch_size,), generator=self.generator, device=device) == 0
        p.masked_fill_(disable_p, 1.0)

        self._compare_results(logits, k, p)
        gc.collect()
        torch.npu.empty_cache()

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_both_disabled(self, device):
        """k=p=None must be a no-op returning the input unchanged."""
        logits = torch.randn(32, 1024, generator=self.generator, dtype=torch.float32, device=device)
        logits_clone = logits.clone()

        result = apply_top_k_top_p_triton(logits_clone, k=None, p=None)

        assert torch.equal(result, logits), "Should be no-op when both k and p are None"

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_k_equals_vocab_noop(self, device):
        """Rows with k == vocab_size keep everything (top-k disabled)."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)

        k = torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert torch.equal(result, logits), "k == vocab_size must be a no-op"

        p = torch.ones(batch_size, dtype=torch.float32, device=device)
        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert torch.equal(result, logits), "p == 1.0 must be a no-op"

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_extreme_values(self, device):
        """Edge cases: k=1, k=V, mixed extremes; p tiny/1.0/mixed."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)

        k = torch.ones(batch_size, dtype=torch.int32, device=device)
        self._compare_results(logits.clone(), k, p=None)

        k = torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
        self._compare_results(logits.clone(), k, p=None)

        k = torch.tensor([1, vocab_size, 2, vocab_size - 1] * 4, dtype=torch.int32, device=device)
        self._compare_results(logits.clone(), k, p=None)

        p = torch.full((batch_size,), 0.01, dtype=torch.float32, device=device)
        self._compare_results(logits.clone(), k=None, p=p)

        p = torch.ones(batch_size, dtype=torch.float32, device=device)
        self._compare_results(logits.clone(), k=None, p=p)

        p = torch.tensor([0.1, 0.5, 0.9, 1.0] * 4, dtype=torch.float32, device=device)
        self._compare_results(logits.clone(), k=None, p=p)

    @pytest.mark.parametrize("mode", ["topk_only", "topp_only", "topk_and_topp"])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_noncontiguous_logits(self, mode, device):
        """Non-contiguous logits views behave like contiguous inputs."""
        batch_size, vocab_size, pad = 16, 4096, 8
        backing = torch.full((batch_size, vocab_size + pad), -1000.0, device=device, dtype=torch.float32)
        base = torch.linspace(10.0, -10.0, vocab_size, device=device, dtype=torch.float32)
        source = base[None, :] + (torch.arange(batch_size, device=device, dtype=torch.float32)[:, None] / 1000.0)

        logits = backing[:, :vocab_size]
        logits.copy_(source)
        contig_logits = source.clone()
        pytorch_logits = source.clone()

        assert logits.stride() == (vocab_size + pad, 1)
        assert not logits.is_contiguous()

        k = None
        p = None
        if mode in ("topk_only", "topk_and_topp"):
            k = torch.full((batch_size,), 154, device=device, dtype=torch.int32)
        if mode in ("topp_only", "topk_and_topp"):
            p = torch.full((batch_size,), 0.95, device=device, dtype=torch.float32)

        noncontig_out = apply_top_k_top_p_triton(logits, k, p)
        contig_out = apply_top_k_top_p_triton(contig_logits, k, p)
        pytorch_out = _apply_top_k_top_p_pytorch(pytorch_logits, k, p)

        assert noncontig_out.data_ptr() == logits.data_ptr()
        assert not noncontig_out.is_contiguous()
        assert torch.equal(logits, noncontig_out)
        assert torch.equal(torch.isfinite(noncontig_out), torch.isfinite(contig_out))
        assert torch.equal(torch.isfinite(noncontig_out), torch.isfinite(pytorch_out))

    # -----------------------------------------------------------------
    # -inf logits (grammar / structured-output bitmask)
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topk_with_neginf_logits(self, inf_fraction, device):
        """Top-k with most logits masked to -inf must not produce NaN."""
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        mask = torch.rand(batch_size, vocab_size, generator=self.generator, device=device) < inf_fraction
        logits[mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, device=device, dtype=torch.int32)
        result = apply_top_k_top_p_triton(logits.clone(), k, None)

        assert not result.isnan().any(), "NaN found in top-k result with -inf logits"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topp_with_neginf_logits(self, inf_fraction, device):
        """Top-p with most logits masked to -inf must not produce NaN."""
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        mask = torch.rand(batch_size, vocab_size, generator=self.generator, device=device) < inf_fraction
        logits[mask] = float("-inf")

        p = torch.rand(batch_size, generator=self.generator, device=device, dtype=torch.float32) * 0.9 + 0.1
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN found in top-p result with -inf logits"
        for i in range(batch_size):
            finite_in = (logits[i] > float("-inf")).sum().item()
            kept = (result[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topk_topp_with_neginf_logits(self, inf_fraction, device):
        """Combined top-k + top-p with most logits masked to -inf."""
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        mask = torch.rand(batch_size, vocab_size, generator=self.generator, device=device) < inf_fraction
        logits[mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, device=device, dtype=torch.int32)
        p = torch.rand(batch_size, generator=self.generator, device=device, dtype=torch.float32) * 0.9 + 0.1
        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), "NaN found in top-k+top-p result with -inf logits"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_all_neginf_logits(self, device):
        """All logits -inf (fully masked grammar): kernel must be a no-op."""
        batch_size, vocab_size = 16, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32, device=device)

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, device=device, dtype=torch.int32)
        p = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert not result.isnan().any(), "NaN from all-inf top-k"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert not result.isnan().any(), "NaN from all-inf top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN from all-inf top-k+top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

    @pytest.mark.parametrize("num_valid", [1, 2, 5, 10, 50])
    @pytest.mark.parametrize("mode", ["topk_only", "topp_only", "topk_and_topp"])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_equal_logits_few_valid(self, num_valid, mode, device):
        """Few valid tokens all sharing the same logit value (grammar pattern).

        The strict `>` keep_mask could exclude everything when the pivot
        converges to max_logit; the guard must keep at least one token.
        """
        batch_size, vocab_size = 32, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32, device=device)
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator, device=device)[:num_valid]
            logits[i, indices] = 1.0

        k = None
        p = None
        if mode in ("topk_only", "topk_and_topp"):
            k = torch.full((batch_size,), max(1, num_valid - 1), dtype=torch.int32, device=device)
        if mode in ("topp_only", "topk_and_topp"):
            p = torch.full((batch_size,), 0.95, dtype=torch.float32, device=device)

        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), "NaN in equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, f"Row {i}: all tokens masked with {num_valid} equal-valued finite logits ({mode})"

    @pytest.mark.parametrize("num_valid", [2, 5, 10])
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_nearly_equal_logits_topp(self, num_valid, device):
        """Few valid tokens with very similar logits (near-degenerate)."""
        batch_size, vocab_size = 32, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32, device=device)
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator, device=device)[:num_valid]
            logits[i, indices] = 1.0 + torch.rand(num_valid, generator=self.generator, device=device) * 1e-6

        p = torch.full((batch_size,), 0.95, dtype=torch.float32, device=device)
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN in nearly-equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, f"Row {i}: all tokens masked with {num_valid} nearly-equal finite logits"

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_mixed_neginf_and_normal_rows(self, device):
        """Batch mixing normal rows with heavily grammar-masked rows."""
        batch_size, vocab_size = 32, 32000
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32, device=device)
        for i in range(0, batch_size, 2):
            mask = torch.rand(vocab_size, generator=self.generator, device=device) < 0.99
            logits[i][mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, device=device, dtype=torch.int32)
        p = torch.rand(batch_size, generator=self.generator, device=device, dtype=torch.float32) * 0.9 + 0.1

        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN in mixed normal/-inf batch"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item()
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept"

    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_topp_tie_break_deterministic(self, device):
        """Boundary-duplicate trimming must be deterministic across calls.

        The tie budget is handed out in index order (tl.cumsum quota), so
        identical calls must produce identical masks.

        Note the kernel and the sort+mask reference differ by design when
        the top-p boundary lands inside an exact tie group: the reference
        keeps the whole group (sort cutoff), the kernel trims it to the
        exact p quota. With 1000 tied tokens of mass ~0.00073 each and
        p=0.5, the kernel keeps 684 while the reference keeps 1000, so
        only the "no larger" direction is asserted here.
        """
        batch_size, vocab_size = 4, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32, device=device)
        logits[:, :1000] = 1.0
        logits[:, 1000:2000] = 0.0
        p = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)

        results = [apply_top_k_top_p_triton(logits.clone(), None, p) for _ in range(5)]
        for r in results[1:]:
            assert torch.equal(results[0], r), "non-deterministic tie break"

        kept_idx = (results[0][0] != float("-inf")).nonzero().flatten()
        assert kept_idx.min().item() == 0
        assert kept_idx.max().item() == len(kept_idx) - 1
        pytorch_kept = (_apply_top_k_top_p_pytorch(logits.clone(), None, p)[0] != float("-inf")).sum().item()
        assert len(kept_idx) <= pytorch_kept, (
            f"kernel kept {len(kept_idx)} tied tokens, reference kept {pytorch_kept}; "
            "the quota trim must never keep more than the whole tie group"
        )

    @pytest.mark.parametrize(
        "batch_size,vocab_size,dtype",
        [
            pytest.param(2048, 155648, torch.float32, id="target_2048x155648"),
        ],
    )
    @pytest.mark.parametrize("device", DEVICES)
    @torch.inference_mode()
    def test_large_shape(self, batch_size, vocab_size, dtype, device):
        """The kernel handles large decode shapes without crashing."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=dtype, device=device)
        k = torch.randint(1, 50, (batch_size,), generator=self.generator, device=device).to(torch.int32)
        p = torch.rand(batch_size, generator=self.generator, device=device) * 0.5 + 0.5

        apply_top_k_top_p_triton(logits, k, p)
        torch.npu.synchronize()

        del logits, k, p
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
