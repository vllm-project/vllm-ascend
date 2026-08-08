# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import Generator
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_ascend.ops.triton.v2.sample.apply_top_k_top_p_triton import apply_top_k_top_p_triton

DEVICE_TYPE = current_platform.device_type


@pytest.fixture(autouse=True)
def reset_default_device():
    """
    Explicitly set the default device, which can affect subsequent tests.
    Adding this fixture helps avoid this problem.
    """
    original_device = torch.get_default_device()
    yield
    torch.set_default_device(original_device)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
class TestTritonTopkTopp:
    """Tests for the Triton top-k/top-p kernel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        torch.set_default_device(DEVICE_TYPE)
        self.generator = Generator(device=DEVICE_TYPE).manual_seed(42)

    def _compare_results(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ):
        """Compare Triton kernel results with the PyTorch reference.

        Top-k only selects logits without arithmetic, so the result must be
        bit-identical to the reference (same kept positions and values).

        For top-p (with or without top-k), the two implementations sum
        probabilities in different orders, so the kept boundary can differ
        slightly. We require:
          - any row whose kept count agrees must keep the SAME positions;
          - kept counts may differ by at most 3 per row, or by < 0.5% of the
            largest kept count (absolute floor first).
        """
        logits_pytorch = logits.clone()
        logits_triton = logits.clone().to(torch.float32)

        result_pytorch = apply_top_k_top_p_pytorch(logits_pytorch, k, p)

        k_i32 = k.to(torch.int32) if k is not None else None
        p_f32 = p.to(torch.float32) if p is not None else None
        result_triton = apply_top_k_top_p_triton(logits_triton, k_i32, p_f32)

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

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topk_only(self, batch_size: int, vocab_size: int):
        """Test top-k only (p=None)."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        k = torch.randint(1, min(100, vocab_size), (batch_size,), generator=self.generator)
        # Randomly disable top-k for some rows (~25%)
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        k.masked_fill_(disable_mask, vocab_size)

        self._compare_results(logits, k, p=None)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topp_only(self, batch_size: int, vocab_size: int):
        """Test top-p only (k=None)."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]
        # Randomly disable top-p for some rows (~25%)
        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        p.masked_fill_(disable_mask, 1.0)

        self._compare_results(logits, k=None, p=p)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
    def test_topk_and_topp(self, batch_size: int, vocab_size: int):
        """Test combined top-k and top-p."""
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        k = torch.randint(1, min(100, vocab_size), (batch_size,), generator=self.generator)
        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]

        # Randomly disable top-k for some rows (~25%)
        disable_k = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        k.masked_fill_(disable_k, vocab_size)
        # Randomly disable top-p for some rows (~25%)
        disable_p = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
        p.masked_fill_(disable_p, 1.0)

        self._compare_results(logits, k, p)

    def test_both_disabled(self):
        """Test when both k and p are None (should be no-op)."""
        logits = torch.randn(32, 1024, generator=self.generator, dtype=torch.float32)
        logits_clone = logits.clone()

        result = apply_top_k_top_p_triton(logits_clone, k=None, p=None)

        assert torch.equal(result, logits), "Should be no-op when both k and p are None"

    def test_extreme_k_values(self):
        """Test edge cases for k values."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)

        # k=1 (keep only top 1)
        k = torch.ones(batch_size, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

        # k=vocab_size (keep all)
        k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

        # Mixed extreme values
        k = torch.tensor([1, vocab_size, 2, vocab_size - 1] * 4, dtype=torch.int32)
        self._compare_results(logits.clone(), k, p=None)

    def test_extreme_p_values(self):
        """Test edge cases for p values."""
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)

        # p close to 0 (very restrictive)
        p = torch.full((batch_size,), 0.01, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

        # p=1.0 (keep all)
        p = torch.ones(batch_size, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

        # Mixed values
        p = torch.tensor([0.1, 0.5, 0.9, 1.0] * 4, dtype=torch.float32)
        self._compare_results(logits.clone(), k=None, p=p)

    def test_large_batch(self):
        """Test with a large batch size."""
        batch_size, vocab_size = 512, 32000
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        k = torch.randint(1, 50, (batch_size,), generator=self.generator)
        p = torch.rand(batch_size, generator=self.generator) * 0.5 + 0.5

        self._compare_results(logits, k, p)

    @pytest.mark.parametrize(
        "mode",
        ["topk_only", "topp_only", "topk_and_topp"],
    )
    def test_noncontiguous_logits_match_contiguous(self, mode: str):
        """Non-contiguous logits views should behave like contiguous inputs."""
        device = torch.device(DEVICE_TYPE)
        batch_size, vocab_size, pad = 16, 4096, 8
        backing = torch.full(
            (batch_size, vocab_size + pad),
            -1000.0,
            device=device,
            dtype=torch.float32,
        )
        base = torch.linspace(10.0, -10.0, vocab_size, device=device, dtype=torch.float32)
        source = base[None, :] + (torch.arange(batch_size, device=device, dtype=torch.float32)[:, None] / 1000.0)

        logits = backing[:, :vocab_size]
        logits.copy_(source)
        contig_logits = source.clone()
        pytorch_logits = source.clone()

        assert logits.shape == (batch_size, vocab_size)
        assert logits.stride() == (vocab_size + pad, 1)
        assert not logits.is_contiguous()

        k: torch.Tensor | None = None
        p: torch.Tensor | None = None
        if mode in ("topk_only", "topk_and_topp"):
            k = torch.full((batch_size,), 154, device=device, dtype=torch.int32)
        if mode in ("topp_only", "topk_and_topp"):
            p = torch.full((batch_size,), 0.95, device=device, dtype=torch.float32)

        noncontig_out = apply_top_k_top_p_triton(logits, k, p)
        contig_out = apply_top_k_top_p_triton(contig_logits, k, p)
        pytorch_out = apply_top_k_top_p_pytorch(pytorch_logits, k, p)

        assert noncontig_out.data_ptr() == logits.data_ptr()
        assert not noncontig_out.is_contiguous()
        assert torch.equal(logits, noncontig_out)
        assert torch.equal(torch.isfinite(noncontig_out), torch.isfinite(contig_out))
        pytorch_kept = torch.isfinite(pytorch_out).sum(dim=-1)
        triton_kept = torch.isfinite(noncontig_out).sum(dim=-1)
        max_diff = (pytorch_kept - triton_kept).abs().max().item()
        max_kept = pytorch_kept.max().item()
        if max_kept > 0 and max_diff > 3:
            assert max_diff / max_kept < 0.005

    # -----------------------------------------------------------------
    # Tests for -inf logits (e.g. from grammar / structured output masks)
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topk_with_neginf_logits(self, inf_fraction: float):
        """Top-k with many -inf logits (simulating grammar bitmask).

        The kernel must not produce NaN when most logits are -inf, which
        can happen when structured-output grammar masks are applied before
        sampling.
        """
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        # Mask a fraction of logits to -inf.
        mask = torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        logits[mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, dtype=torch.int32)
        result = apply_top_k_top_p_triton(logits.clone(), k, None)

        assert not result.isnan().any(), "NaN found in top-k result with -inf logits"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"
            # At least one value should survive unless the row was all -inf.
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topp_with_neginf_logits(self, inf_fraction: float):
        """Top-p with many -inf logits."""
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        mask = torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        logits[mask] = float("-inf")

        p = torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9 + 0.1
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN found in top-p result with -inf logits"
        for i in range(batch_size):
            finite_in = (logits[i] > float("-inf")).sum().item()
            kept = (result[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept despite finite input"

    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
    def test_topk_topp_with_neginf_logits(self, inf_fraction: float):
        """Combined top-k + top-p with many -inf logits."""
        batch_size, vocab_size = 32, 128256
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        mask = torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
        logits[mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, dtype=torch.int32)
        p = torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9 + 0.1
        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), "NaN found in top-k+top-p result with -inf logits"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"

    def test_all_neginf_logits(self):
        """All logits are -inf (fully masked). Kernel should be a no-op."""
        batch_size, vocab_size = 16, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32)

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, dtype=torch.int32)
        p = torch.full((batch_size,), 0.9, dtype=torch.float32)

        # top-k only
        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert not result.isnan().any(), "NaN from all-inf top-k"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        # top-p only
        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert not result.isnan().any(), "NaN from all-inf top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

        # top-k + top-p
        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN from all-inf top-k+top-p"
        assert (result == float("-inf")).all(), "Expected all -inf unchanged"

    def test_few_valid_tokens_with_neginf(self):
        """Only a handful of tokens are finite per row (strict grammar)."""
        batch_size, vocab_size = 32, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32)
        # Allow only 5 random tokens per row to be finite.
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:5]
            logits[i, indices] = torch.randn(5, generator=self.generator, dtype=torch.float32)

        k = torch.full((batch_size,), 50, dtype=torch.int32)
        p = torch.full((batch_size,), 0.9, dtype=torch.float32)

        # top-k only (k=50 but only 5 finite → keep all 5)
        result = apply_top_k_top_p_triton(logits.clone(), k, None)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept == 5, f"Row {i}: expected 5 kept, got {kept}"

        # top-k with k < num_finite
        k_small = torch.full((batch_size,), 3, dtype=torch.int32)
        result = apply_top_k_top_p_triton(logits.clone(), k_small, None)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= 3, f"Row {i}: expected <=3 kept, got {kept}"

        # top-p only
        result = apply_top_k_top_p_triton(logits.clone(), None, p)
        assert not result.isnan().any()
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, f"Row {i}: no tokens kept"

    @pytest.mark.parametrize("num_valid", [1, 2, 5, 10, 50])
    @pytest.mark.parametrize(
        "mode",
        ["topk_only", "topp_only", "topk_and_topp"],
    )
    def test_equal_logits_few_valid(self, num_valid: int, mode: str):
        """Few valid tokens all sharing the same logit value.

        This is the pattern produced by grammar bitmask filtering when
        the model assigns similar scores to the few allowed tokens.
        The ternary search can converge to a pivot equal to max_logit,
        causing the strict `>` keep_mask to exclude everything.
        Regression test for the `final_pivot >= max_logit` guard.
        """
        batch_size, vocab_size = 32, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32)
        # Set exactly `num_valid` tokens per row to the SAME finite value.
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
            logits[i, indices] = 1.0  # all equal

        k: torch.Tensor | None = None
        p: torch.Tensor | None = None
        if mode in ("topk_only", "topk_and_topp"):
            k = torch.full((batch_size,), max(1, num_valid - 1), dtype=torch.int32)
        if mode in ("topp_only", "topk_and_topp"):
            p = torch.full((batch_size,), 0.95, dtype=torch.float32)

        result = apply_top_k_top_p_triton(logits.clone(), k, p)

        assert not result.isnan().any(), "NaN in equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            # The key invariant: at least one token must survive.
            # With all-equal logits the pivot search can't differentiate
            # tokens, so the guard may keep more than k — that is the
            # intended safe fallback.
            assert kept > 0, f"Row {i}: all tokens masked with {num_valid} equal-valued finite logits ({mode})"

    @pytest.mark.parametrize("num_valid", [2, 5, 10])
    def test_nearly_equal_logits_topp(self, num_valid: int):
        """Few valid tokens with very similar (but not identical) logits.

        Ensures the kernel handles near-degenerate probability
        distributions where the ternary search range collapses.
        """
        batch_size, vocab_size = 32, 128256
        logits = torch.full((batch_size, vocab_size), float("-inf"), dtype=torch.float32)
        for i in range(batch_size):
            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
            # Tiny spread: values in [1.0, 1.0 + 1e-6]
            logits[i, indices] = 1.0 + torch.rand(num_valid, generator=self.generator, dtype=torch.float32) * 1e-6

        p = torch.full((batch_size,), 0.95, dtype=torch.float32)
        result = apply_top_k_top_p_triton(logits.clone(), None, p)

        assert not result.isnan().any(), "NaN in nearly-equal-logit result"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept > 0, f"Row {i}: all tokens masked with {num_valid} nearly-equal finite logits"

    def test_mixed_neginf_and_normal_rows(self):
        """Batch with a mix of normal rows and heavily-masked rows."""
        batch_size, vocab_size = 32, 32000
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        # Mask even rows heavily (99% -inf), leave odd rows normal.
        for i in range(0, batch_size, 2):
            mask = torch.rand(vocab_size, generator=self.generator) < 0.99
            logits[i][mask] = float("-inf")

        k = torch.randint(1, 50, (batch_size,), generator=self.generator, dtype=torch.int32)
        p = torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9 + 0.1

        result = apply_top_k_top_p_triton(logits.clone(), k, p)
        assert not result.isnan().any(), "NaN in mixed normal/-inf batch"
        for i in range(batch_size):
            kept = (result[i] > float("-inf")).sum().item()
            assert kept <= k[i].item()
            finite_in = (logits[i] > float("-inf")).sum().item()
            if finite_in > 0:
                assert kept > 0, f"Row {i}: no tokens kept"

    # -----------------------------------------------------------------
    # Additional input-contract tests
    # -----------------------------------------------------------------

    def test_k_zero(self):
        """k=0 degenerates to keeping only the maximum logit.

        The PyTorch reference treats k=0 as a top-k boundary equal to the
        row max: every strictly smaller logit is masked, so exactly the
        argmax survives.  The Triton kernel must match that exactly
        (positions and values) without producing NaN.
        """
        batch_size, vocab_size = 16, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        k = torch.zeros(batch_size, dtype=torch.int32)

        result = apply_top_k_top_p_triton(logits.clone(), k, None)

        assert not result.isnan().any(), "NaN found with k=0"
        # The degenerate top-k boundary keeps at least the argmax per row.
        kept = (result > float("-inf")).sum(dim=-1)
        assert kept.min().item() >= 1, "k=0 should keep at least the argmax"

        # Exact equivalence with the reference (same kept positions/values).
        self._compare_results(logits.clone(), k, None)

    def test_empty_batch(self):
        """batch_size == 0 must be a no-op returning the input unchanged."""
        logits = torch.randn(0, 1024, generator=self.generator, dtype=torch.float32)
        k = torch.empty(0, dtype=torch.int32)
        p = torch.empty(0, dtype=torch.float32)

        out = apply_top_k_top_p_triton(logits, k, p)
        assert out is logits
        assert out.shape == (0, 1024)

        # Both disabled is also a no-op.
        out2 = apply_top_k_top_p_triton(logits, None, None)
        assert out2 is logits

    def test_custom_mask_value(self):
        """mask_value fills masked positions; kept positions/values unchanged."""
        batch_size, vocab_size = 32, 8192
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)
        k = torch.randint(1, 100, (batch_size,), generator=self.generator)
        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1
        mask_value = -100.0

        out_custom = apply_top_k_top_p_triton(logits.clone(), k, p, mask_value=mask_value)
        out_default = apply_top_k_top_p_triton(logits.clone(), k, p)

        custom_mask = out_custom == mask_value
        default_mask = out_default == float("-inf")
        assert torch.equal(custom_mask, default_mask), "custom mask_value changed the masked positions"
        assert torch.equal(out_custom[~custom_mask], out_default[~default_mask]), (
            "custom mask_value changed the kept values"
        )

    def test_invalid_inputs_rejected(self):
        """Invalid shapes / dtypes must raise AssertionError."""
        batch_size, vocab_size = 8, 1024
        logits = torch.randn(batch_size, vocab_size, generator=self.generator, dtype=torch.float32)

        # Non-float32 logits.
        with pytest.raises(AssertionError):
            apply_top_k_top_p_triton(logits.to(torch.float16), None, None)
        # Non-2D logits.
        with pytest.raises(AssertionError):
            apply_top_k_top_p_triton(logits.unsqueeze(0), None, None)
        # k length mismatch.
        with pytest.raises(AssertionError):
            apply_top_k_top_p_triton(logits, torch.ones(batch_size + 1, dtype=torch.int32), None)
        # k wrong ndim.
        with pytest.raises(AssertionError):
            apply_top_k_top_p_triton(logits, torch.ones(1, batch_size, dtype=torch.int32), None)
        # p length mismatch.
        with pytest.raises(AssertionError):
            apply_top_k_top_p_triton(logits, None, torch.ones(batch_size + 1, dtype=torch.float32))
