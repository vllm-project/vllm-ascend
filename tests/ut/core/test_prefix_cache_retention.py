#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the SWA prefix-cache retention three-state (vLLM PR #43447).

These cover the parts that the DSv4 prefix-cache patch backports onto vLLM
v0.21.0:

* ``SlidingWindowManager.reachable_block_mask`` (the pure mask algorithm,
  including the deliberate deviation that env-unset / ``None`` keeps the dense
  cache-all behavior),
* ``BlockPool.cache_full_blocks(block_mask=...)`` (the null-swap mask plumbing
  that v0.21.0 lacks),
* ``_validate_prefix_cache_retention_interval`` (the no-SWA-group / negative /
  non-multiple failure modes, incl. the SlidingWindowMLASpec acceptance fix).

The patches attach methods onto the upstream vLLM classes, so importing this
module requires vLLM (as on the CI / NPU environment); ``TestBase`` applies the
ascend patches for us.
"""
from types import SimpleNamespace

from tests.ut.base import TestBase
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    _SLIDING_WINDOW_SPECS,
    _validate_prefix_cache_retention_interval,
)


def _swa_spec(block_size: int, sliding_window: int) -> SlidingWindowSpec:
    """Build a minimal ``SlidingWindowSpec`` instance.

    ``reachable_block_mask`` only reads ``block_size`` and ``sliding_window`` and
    asserts ``isinstance(spec, SlidingWindowSpec)``. ``SlidingWindowSpec`` is a
    frozen dataclass with many required fields, so we bypass ``__init__`` and set
    just the two attributes the algorithm touches.
    """
    spec = object.__new__(SlidingWindowSpec)
    object.__setattr__(spec, "block_size", block_size)
    object.__setattr__(spec, "sliding_window", sliding_window)
    return spec


def _mask(
    *,
    start_block,
    end_block,
    alignment_tokens,
    block_size,
    sliding_window,
    use_eagle=False,
    retention_interval=None,
    num_prompt_tokens=None,
):
    return SlidingWindowManager.reachable_block_mask(
        start_block=start_block,
        end_block=end_block,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=_swa_spec(block_size, sliding_window),
        use_eagle=use_eagle,
        retention_interval=retention_interval,
        num_prompt_tokens=num_prompt_tokens,
    )


def _cached_set(mask, start_block=0):
    """Indices (absolute block ids) whose mask entry is True."""
    return {start_block + i for i, keep in enumerate(mask) if keep}


class TestReachableBlockMask(TestBase):
    """Pure-function coverage of the three-state mask algorithm."""

    def test_default_none_caches_all_swa(self):
        # Deviation from PR #43447 + our hard requirement: env unset (None) must
        # return None so cache_full_blocks caches every block (dense cache-all).
        mask = _mask(
            start_block=0,
            end_block=16,
            alignment_tokens=128,
            block_size=8,
            sliding_window=8,
            retention_interval=None,
            num_prompt_tokens=128,
        )
        self.assertIsNone(mask)

    def test_dense_equivalence_path_ignores_alignment(self):
        # The dense safety net (None) must not depend on scheduler_block_size:
        # even with alignment_tokens=None it returns None.
        mask = _mask(
            start_block=0,
            end_block=8,
            alignment_tokens=None,
            block_size=8,
            sliding_window=8,
            retention_interval=None,
            num_prompt_tokens=64,
        )
        self.assertIsNone(mask)

    def test_interval_64_segment_tails_plus_replay(self):
        # interval=64, block_size=8 -> per_segment=8; sliding_window=8 -> need=1.
        # Each 8-block segment keeps only its last block (offset 7 within the
        # segment), plus the replay tail near the latest prompt boundary.
        mask = _mask(
            start_block=0,
            end_block=16,
            alignment_tokens=64,
            block_size=8,
            sliding_window=8,
            retention_interval=64,
            num_prompt_tokens=128,
        )
        # Segment tails: block 7 (segment 0) and block 15 (segment 1).
        # latest = (128-1)//64*64 = 64 -> prompt_end_block = 64//8 = 8 ->
        # replay range [8-1, 8) = {7}. So cached = {7, 15}.
        self.assertEqual(_cached_set(mask), {7, 15})

    def test_latest_only_zero(self):
        # retention=0 -> no per-segment tails, only the replay tail survives.
        mask = _mask(
            start_block=0,
            end_block=16,
            alignment_tokens=64,
            block_size=8,
            sliding_window=8,
            retention_interval=0,
            num_prompt_tokens=128,
        )
        # latest = 64, prompt_end_block = 8, need = 1 -> replay {7}.
        self.assertEqual(_cached_set(mask), {7})

    def test_eagle_zero(self):
        # use_eagle=True, 127 tokens, block_size=8, window=8 -> need=2, shift=1,
        # latest=96, prompt_end_block=96//8 + 1 = 13 -> cached={11,12}.
        mask = _mask(
            start_block=0,
            end_block=16,
            alignment_tokens=32,
            block_size=8,
            sliding_window=8,
            use_eagle=True,
            retention_interval=0,
            num_prompt_tokens=127,
        )
        self.assertEqual(_cached_set(mask), {11, 12})

    def test_need_ge_per_segment_folds_to_none(self):
        # window large enough that need >= per_segment: the whole segment is in
        # reach, so nothing can be dropped -> dense fallback (None).
        mask = _mask(
            start_block=0,
            end_block=16,
            alignment_tokens=16,
            block_size=8,
            sliding_window=64,  # need = cdiv(63, 8) = 8 >= per_segment(=2)
            retention_interval=16,
            num_prompt_tokens=128,
        )
        self.assertIsNone(mask)

    def test_start_block_offset_respected(self):
        # Already-cached prefix (start_block>0): mask length == end-start and
        # indices stay absolute. interval=64 -> per_segment=8, need=1.
        mask = _mask(
            start_block=8,
            end_block=16,
            alignment_tokens=64,
            block_size=8,
            sliding_window=8,
            retention_interval=64,
            num_prompt_tokens=128,
        )
        self.assertEqual(len(mask), 8)
        # Segment-1 tail is block 15; replay tail (latest=64 -> end_block 8) is
        # below start_block=8, so only {15} remains.
        self.assertEqual(_cached_set(mask, start_block=8), {15})


class TestCacheFullBlocksMask(TestBase):
    """Directly exercise the BlockPool.cache_full_blocks(block_mask=...) patch."""

    def _make_block_pool(self):
        from vllm.v1.core.block_pool import BlockPool

        # Small real pool: caching enabled, no KV events.
        return BlockPool(
            num_gpu_blocks=16,
            enable_caching=True,
            hash_block_size=8,
            enable_kv_cache_events=False,
        )

    def _request(self, num_blocks, block_size):
        # cache_full_blocks reads request.block_hashes / all_token_ids; provide
        # enough distinct chained hashes for num_blocks full blocks.
        block_hashes = [bytes([i]) * 4 for i in range(num_blocks + 2)]
        return SimpleNamespace(
            request_id="req-mask",
            block_hashes=block_hashes,
            all_token_ids=list(range(num_blocks * block_size)),
            lora_request=None,
        )

    def test_mask_caches_only_unmasked_and_restores_blocks(self):
        block_pool = self._make_block_pool()
        block_size = 8
        blocks = block_pool.get_new_blocks(3)
        original = list(blocks)
        request = self._request(num_blocks=3, block_size=block_size)

        block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=3,
            block_size=block_size,
            kv_cache_group_id=0,
            block_mask=[True, False, True],
        )

        # Unmasked blocks (0, 2) get a block_hash; masked block (1) stays None.
        self.assertIsNotNone(blocks[0].block_hash)
        self.assertIsNone(blocks[1].block_hash)
        self.assertIsNotNone(blocks[2].block_hash)
        # The request's block list is restored verbatim (no null pollution).
        self.assertEqual(blocks, original)

    def test_mask_none_caches_all(self):
        block_pool = self._make_block_pool()
        block_size = 8
        blocks = block_pool.get_new_blocks(2)
        request = self._request(num_blocks=2, block_size=block_size)

        block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=2,
            block_size=block_size,
            kv_cache_group_id=0,
            block_mask=None,
        )

        self.assertIsNotNone(blocks[0].block_hash)
        self.assertIsNotNone(blocks[1].block_hash)


class TestValidateRetentionInterval(TestBase):
    """Validation failure modes (vLLM PR #43447 (3))."""

    def _config_with_specs(self, specs):
        groups = [SimpleNamespace(kv_cache_spec=s) for s in specs]
        return SimpleNamespace(kv_cache_groups=groups)

    def test_none_is_noop(self):
        cfg = self._config_with_specs([SimpleNamespace()])  # no SWA group
        # None must never raise, even without an SWA group.
        _validate_prefix_cache_retention_interval(None, 128, cfg)

    def test_reject_no_sliding_window_group(self):
        cfg = self._config_with_specs([SimpleNamespace()])
        with self.assertRaisesRegex(ValueError, "no sliding-window KV cache group"):
            _validate_prefix_cache_retention_interval(64, 64, cfg)

    def test_reject_non_multiple(self):
        cfg = self._config_with_specs([_swa_spec(8, 8)])
        with self.assertRaisesRegex(ValueError, "multiple of scheduler_block_size"):
            _validate_prefix_cache_retention_interval(33, 32, cfg)

    def test_reject_negative(self):
        cfg = self._config_with_specs([_swa_spec(8, 8)])
        # -32 % 32 == 0 but must still be rejected by the explicit < 0 check.
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_prefix_cache_retention_interval(-32, 32, cfg)

    def test_mla_swa_spec_accepted(self):
        # DSv4 SWA is SlidingWindowMLASpec; the validator must treat it as an
        # SWA group (else DSv4 + retention wrongly reports "no SWA group").
        mla_spec_cls = _SLIDING_WINDOW_SPECS[-1]
        spec = object.__new__(mla_spec_cls)
        object.__setattr__(spec, "block_size", 128)
        object.__setattr__(spec, "sliding_window", 128)
        cfg = self._config_with_specs([spec])
        # A valid multiple should not raise for an MLA-SWA group.
        _validate_prefix_cache_retention_interval(128, 128, cfg)
