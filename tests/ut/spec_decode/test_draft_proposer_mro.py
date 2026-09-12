#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Regression tests for MRO resolution of ``_maybe_share_lm_head``.

``AscendDraftModelProposer`` inherits from upstream ``DraftModelProposer``
first and ``AscendSpecDecodeBaseProposer`` second. Upstream's
``_maybe_share_lm_head`` is a deliberate no-op (independent draft models
don't share lm_head), so when it wins MRO resolution it silently skips the
Ascend implementation -- which, for the draft_model method, is the only
place that arms ACL full-graph support (``self.update_stream`` and the
``ACLGraphWrapper``-wrapped ``self._runnable``). The subclass therefore
overrides the method and delegates explicitly to the Ascend base; these
tests pin that contract.
"""

from __future__ import annotations

from unittest import mock

from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def test_maybe_share_lm_head_not_shadowed_by_upstream_no_op():
    # The upstream no-op used to win MRO resolution, making the drafter
    # crash with "AttributeError: ... no attribute 'update_stream'" in
    # cudagraph mode (and silently skip draft graph capture).
    resolved = AscendDraftModelProposer._maybe_share_lm_head
    assert resolved is not DraftModelProposer._maybe_share_lm_head


def test_maybe_share_lm_head_delegates_to_ascend_base():
    fake_self = object()
    fake_model = object()
    with mock.patch.object(AscendSpecDecodeBaseProposer, "_maybe_share_lm_head", autospec=True) as base_impl:
        # Unbound-method style call with a stand-in instance; mypy cannot
        # see that autospec makes the receiver type irrelevant here.
        AscendDraftModelProposer._maybe_share_lm_head(fake_self, fake_model)  # type: ignore[arg-type]
    base_impl.assert_called_once_with(fake_self, fake_model)


def test_mro_order_unchanged_by_the_fix():
    # The fix must delegate only ``_maybe_share_lm_head``; the class MRO is
    # unchanged, so unrelated methods still resolve upstream-first.
    assert AscendDraftModelProposer.__mro__[:3] == (
        AscendDraftModelProposer,
        DraftModelProposer,
        AscendSpecDecodeBaseProposer,
    )
