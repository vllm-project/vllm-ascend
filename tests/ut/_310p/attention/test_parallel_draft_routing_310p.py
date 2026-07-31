#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import contextlib
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackendImpl310
from vllm_ascend.attention.attention_v1 import AscendAttentionState

ATTN_MOD = "vllm_ascend._310p.attention.attention_v1"

CAUSAL_STATES = [
    AscendAttentionState.PrefillNoCache,
    AscendAttentionState.DecodeOnly,
    AscendAttentionState.ChunkedPrefill,
    AscendAttentionState.PrefillCacheHit,
    AscendAttentionState.SpecDecoding,
]


def make_impl(method="dflash"):
    impl = AscendAttentionBackendImpl310.__new__(AscendAttentionBackendImpl310)
    impl.vllm_config = SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method) if method is not None else None)
    )
    impl._fia_scope_validated = False
    return impl


def make_metadata(state, causal):
    return SimpleNamespace(attn_state=state, causal=causal)


@contextlib.contextmanager
def routed(is_draft_model):
    """Patch the three seams forward_impl dispatches on, and record which fired."""
    calls = {}

    def record(name):
        def fn(*args, **kwargs):
            calls[name] = calls.get(name, 0) + 1
            return name

        return fn

    with (
        mock_patch(f"{ATTN_MOD}._EXTRA_CTX", SimpleNamespace(is_draft_model=is_draft_model)),
        mock_patch(f"{ATTN_MOD}.forward_parallel_draft_fia", record("fia")),
        mock_patch.object(AscendAttentionBackendImpl310, "forward_prefill_310", record("prefill")),
        mock_patch.object(AscendAttentionBackendImpl310, "forward_paged_attention", record("paged")),
        mock_patch.object(AscendAttentionBackendImpl310, "forward_chunked_prefill_310", record("splitfuse")),
    ):
        yield calls


def call(impl, md):
    return impl.forward_impl(None, None, None, None, md, None)


class TestParallelDraftRouting(TestBase):
    def test_dflash_draft_non_causal_chunked_prefill_goes_to_adn(self):
        with routed(is_draft_model=True) as calls:
            result = call(make_impl("dflash"), make_metadata(AscendAttentionState.ChunkedPrefill, causal=False))
        self.assertEqual(result, "fia")
        self.assertEqual(calls, {"fia": 1})

    def test_dspark_draft_non_causal_chunked_prefill_goes_to_adn(self):
        with routed(is_draft_model=True) as calls:
            result = call(make_impl("dspark"), make_metadata(AscendAttentionState.ChunkedPrefill, causal=False))
        self.assertEqual(result, "fia")
        self.assertEqual(calls, {"fia": 1})

    def test_causal_paths_are_unchanged(self):
        """Every causal state must keep its existing 310P route, custom FIA untouched."""
        expected = {
            AscendAttentionState.PrefillNoCache: "prefill",
            AscendAttentionState.DecodeOnly: "paged",
            AscendAttentionState.ChunkedPrefill: "splitfuse",
            AscendAttentionState.PrefillCacheHit: "splitfuse",
            AscendAttentionState.SpecDecoding: "splitfuse",
        }
        for state in CAUSAL_STATES:
            for is_draft in (False, True):
                with routed(is_draft_model=is_draft) as calls:
                    result = call(make_impl("dflash"), make_metadata(state, causal=True))
                self.assertEqual(result, expected[state], f"{state} (draft={is_draft}) took the wrong route")
                self.assertNotIn("fia", calls, f"{state} reached custom FIA despite being causal")

    def test_non_draft_non_causal_fails_loud(self):
        """Target attention must never silently fall through to the causal kernel."""
        with routed(is_draft_model=False) as calls:
            with self.assertRaisesRegex(NotImplementedError, "no non-causal attention path"):
                call(make_impl("dflash"), make_metadata(AscendAttentionState.ChunkedPrefill, causal=False))
        self.assertEqual(calls, {})

    def test_unsupported_method_non_causal_fails_loud(self):
        with routed(is_draft_model=True) as calls:
            with self.assertRaisesRegex(NotImplementedError, "no non-causal attention path"):
                call(make_impl("eagle3"), make_metadata(AscendAttentionState.ChunkedPrefill, causal=False))
        self.assertEqual(calls, {})

    def test_no_speculative_config_non_causal_fails_loud(self):
        with routed(is_draft_model=True) as calls:
            with self.assertRaisesRegex(NotImplementedError, "no non-causal attention path"):
                call(make_impl(method=None), make_metadata(AscendAttentionState.ChunkedPrefill, causal=False))
        self.assertEqual(calls, {})

    def test_causal_paths_do_not_need_a_forward_context(self):
        """Causal routing must not consult _EXTRA_CTX.

        Reading it requires an active forward context. Adding that requirement to
        the causal path would break callers that never needed one -- as it did
        break test_forward_mtp_310, which mocks out the split-fuse method and so
        establishes no context.
        """

        class Exploding:
            def __getattr__(self, name):
                raise AssertionError(f"causal path read _EXTRA_CTX.{name}")

        for state in CAUSAL_STATES:
            with (
                mock_patch(f"{ATTN_MOD}._EXTRA_CTX", Exploding()),
                mock_patch(f"{ATTN_MOD}.forward_parallel_draft_fia"),
                mock_patch.object(AscendAttentionBackendImpl310, "forward_prefill_310", lambda *a: "prefill"),
                mock_patch.object(AscendAttentionBackendImpl310, "forward_paged_attention", lambda *a: "paged"),
                mock_patch.object(
                    AscendAttentionBackendImpl310, "forward_chunked_prefill_310", lambda *a: "splitfuse"
                ),
            ):
                call(make_impl("dflash"), make_metadata(state, causal=True))

    def test_draft_non_causal_in_other_states_fails_loud(self):
        """Only ChunkedPrefill is validated; other non-causal states must refuse."""
        for state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding):
            with routed(is_draft_model=True) as calls:
                with self.assertRaisesRegex(NotImplementedError, "no non-causal attention path"):
                    call(make_impl("dflash"), make_metadata(state, causal=False))
            self.assertEqual(calls, {}, f"{state} should not have been routed anywhere")
