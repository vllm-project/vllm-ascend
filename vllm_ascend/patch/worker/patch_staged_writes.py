# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/buffer_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Skip the no-op staged-write flush of BlockTables.num_blocks. On platforms
# without mapped pinned host memory (the UVA fallback path), every
# copy_to_uva issues a real H2D copy with per-call launch overhead.
# num_blocks.np is written only in append_block_ids, so a decode step that
# appended no blocks pays one wasted H2D copy per step.
#
# The equivalent Sampler flush guard is already upstream (vllm#62748) and
# arrives through main2main, so it is not patched here.
from vllm.v1.worker.gpu.block_table import BlockTables

_orig_block_tables_init = BlockTables.__init__
_orig_append_block_ids = BlockTables.append_block_ids


def _block_tables_init(self, *args, **kwargs):
    _orig_block_tables_init(self, *args, **kwargs)
    self._num_blocks_dirty = False
    orig_num_blocks_copy = self.num_blocks.copy_to_uva

    def copy_to_uva(n=None):
        if not self._num_blocks_dirty:
            return self.num_blocks.gpu
        copied = orig_num_blocks_copy(n)
        # Clear only after a successful copy so a failed flush is retried on
        # the next step instead of leaving the device buffer stale.
        self._num_blocks_dirty = False
        return copied

    self.num_blocks.copy_to_uva = copy_to_uva


def _append_block_ids(self, req_index, new_block_ids, overwrite):
    _orig_append_block_ids(self, req_index, new_block_ids, overwrite)
    self._num_blocks_dirty = True


BlockTables.__init__ = _block_tables_init
BlockTables.append_block_ids = _append_block_ids


# ---------------------------------------------------------------------------
# Guard sampler state copy_to_uva() calls so decode-only steps do not issue
# redundant H2D copies when no request metadata changed.
# ---------------------------------------------------------------------------


def _guard_state_class(cls, field_specs: tuple[tuple[str, str], ...]) -> None:
    if getattr(cls, "_ascend_dirty_patched", False):
        return
    cls._ascend_dirty_patched = True

    orig_init = cls.__init__
    orig_add_request = cls.add_request

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Skip wrapping if upstream already implements dirty tracking on this class.
        if any(hasattr(self, dirty_attr) for _, dirty_attr in field_specs):
            return
        for field_name, dirty_attr in field_specs:
            setattr(self, dirty_attr, False)
            tensor = getattr(self, field_name)
            orig_copy = tensor.copy_to_uva

            def _make_guarded_copy(_orig=orig_copy, _tensor=tensor, _dirty_attr=dirty_attr):
                def _guarded_copy(n=None):
                    if not getattr(self, _dirty_attr, True):
                        return _tensor.gpu
                    copied = _orig(n)
                    setattr(self, _dirty_attr, False)
                    return copied

                return _guarded_copy

            tensor.copy_to_uva = _make_guarded_copy()

    def _patched_add_request(self, req_idx: int, sampling_params, *args, **kwargs):
        if not any(hasattr(self, dirty_attr) for _, dirty_attr in field_specs):
            return orig_add_request(self, req_idx, sampling_params, *args, **kwargs)
        prev_vals = [getattr(self, field_name).np[req_idx] for field_name, _ in field_specs]
        ret = orig_add_request(self, req_idx, sampling_params, *args, **kwargs)
        skip_seed = (
            getattr(sampling_params, "seed", None) is None and getattr(sampling_params, "temperature", None) == 0.0
        )
        for (field_name, dirty_attr), prev in zip(field_specs, prev_vals):
            arr = getattr(self, field_name).np
            if field_name == "seeds" and skip_seed:
                arr[req_idx] = prev
                continue
            if arr[req_idx] != prev:
                setattr(self, dirty_attr, True)
        return ret

    cls.__init__ = _patched_init
    cls.add_request = _patched_add_request


def _patch_sampler_dirty_guards() -> None:
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.logprob import LogprobState
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    _guard_state_class(BadWordsState, (("num_bad_words", "_ascend_bad_words_dirty"),))
    _guard_state_class(
        PenaltiesState,
        (
            ("repetition_penalty", "_ascend_rep_dirty"),
            ("frequency_penalty", "_ascend_freq_dirty"),
            ("presence_penalty", "_ascend_pres_dirty"),
        ),
    )
    _guard_state_class(
        LogitBiasState,
        (
            ("num_allowed_token_ids", "_ascend_allowed_dirty"),
            ("num_logit_bias", "_ascend_bias_dirty"),
            ("min_lens", "_ascend_min_lens_dirty"),
            ("num_stop_token_ids", "_ascend_stop_dirty"),
            ("restore_when_all_masked", "_ascend_restore_dirty"),
        ),
    )
    _guard_state_class(LogprobState, (("num_token_ids", "_ascend_num_token_ids_dirty"),))
    _guard_state_class(
        SamplingStates,
        (
            ("temperature", "_ascend_temp_dirty"),
            ("top_p", "_ascend_top_p_dirty"),
            ("top_k", "_ascend_top_k_dirty"),
            ("min_p", "_ascend_min_p_dirty"),
            ("seeds", "_ascend_seeds_dirty"),
        ),
    )


_patch_sampler_dirty_guards()
