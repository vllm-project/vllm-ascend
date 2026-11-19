#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

"""Python adapter over the in-tree ``_pattern_native`` extension."""

from __future__ import annotations

import operator

import numpy as np

from vllm_ascend.observability.runtime_guard.native_pattern._pattern_native import Detector
from vllm_ascend.observability.runtime_guard.native_pattern.config import Result
from vllm_ascend.observability.runtime_guard.native_pattern.rules import load_pattern_rules


class PatternDetector:
    """Batched pattern-repeat detector backed by the native CPU extension."""

    def __init__(self, tables, think_end=None, *, rules_path: str | None = None):
        if think_end is not None:
            # int(float) silently changes token identity. Accept integer-like
            # scalars only; None explicitly disables the answer boundary.
            if isinstance(think_end, (bool, np.bool_)):
                raise TypeError("think_end must be an integer token ID or None")
            think_end = operator.index(think_end)
            if not 0 <= think_end <= np.iinfo(np.int64).max:
                raise ValueError("think_end must be a non-negative int64 token ID")
        columns = ("norm", "chars", "ws", "run_char", "marker", "send")
        arrays = []
        size = None
        bounds = np.iinfo(np.int32)
        for key in columns:
            array = np.asarray(tables[key])
            if array.ndim != 1 or not array.size:
                raise ValueError(f"{key} must be a non-empty one-dimensional table")
            if size is not None and array.size != size:
                raise ValueError("vocabulary columns must have equal lengths")
            size = array.size
            if array.dtype.kind not in "biu":
                raise TypeError(f"{key} must contain integer or boolean values")
            if int(array.min()) < bounds.min or int(array.max()) > bounds.max:
                raise ValueError(f"{key} values must fit int32 without truncation")
            arrays.append(np.ascontiguousarray(array, dtype=np.int32))
        rules = load_pattern_rules(rules_path)
        self.titles = rules["titles"]
        self.reasons = rules["reasons"]
        self.native = Detector(
            *arrays,
            -1 if think_end is None else think_end,
            rules["exact_params"],
            int(rules["run_max_threshold"]),
            rules["feature_windows"],
            rules["atom_meta"],
            rules["atom_thr"],
            rules["persist"],
        )

    def start(self):
        return self.native.start()

    def result(self, st):
        if st.failed:
            raise RuntimeError("failed detector state; discard it")
        if st.reason < 0:
            return None
        return Result(True, st.fired_at, self.titles[st.reason], self.reasons[st.reason])

    def progress(self, st):
        """Exceptional-path receipt, not a retry/rollback promise.

        A batch can commit earlier slots before a later slot raises. Query each
        state to recover its committed prefix/first alert without consuming any
        token. A failed native block has no trustworthy partial counters.
        """
        if st.failed:
            return None
        title = self.titles[st.reason] if st.reason >= 0 else ""
        rule = self.reasons[st.reason] if st.reason >= 0 else ""
        return (st.received, st.t, st.fired_at, title, rule)

    def feed_events(self, states, token_lists):
        """Return ``(slot, fired_at, title, rule)`` for new alerts only."""
        return [
            (slot, fired_at, self.titles[reason], self.reasons[reason])
            for slot, fired_at, reason in self.native.feed_events(states, token_lists)
        ]

    def finish(self, st):
        self.native.finish(st)
        return self.result(st) or Result()
