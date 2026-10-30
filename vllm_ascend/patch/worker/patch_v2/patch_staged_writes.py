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
# Skip staged-write flushes that carry no data change. On platforms without
# mapped pinned host memory (the UVA fallback path), every copy_to_uva issues
# a real H2D copy with per-call launch overhead; on steps where nothing was
# staged these copies are pure waste.
#
# 1. `vllm.v1.worker.gpu.sample.sampler.Sampler.apply_staged_writes`
#    All sampler sub-states (sampling params, logit bias, bad words,
#    penalties, logprobs, thinking budget, trace replay) stage data only in
#    their add_request, which is only called from Sampler.add_request when a
#    new request is scheduled. Skip the flush entirely on steps without new
#    requests (~10 copy_to_uva per decode step).
# 2. `vllm.v1.worker.gpu.block_table.BlockTables` num_blocks flush
#    num_blocks.np is written only in append_block_ids; flush the UVA-backed
#    tensor only when at least one row changed (1 copy per decode step
#    otherwise).
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.sample.sampler import Sampler

_orig_block_tables_init = BlockTables.__init__
_orig_append_block_ids = BlockTables.append_block_ids
_orig_sampler_add_request = Sampler.add_request
_orig_sampler_apply_staged_writes = Sampler.apply_staged_writes


def _block_tables_init(self, *args, **kwargs):
    _orig_block_tables_init(self, *args, **kwargs)
    self._num_blocks_dirty = False
    orig_num_blocks_copy = self.num_blocks.copy_to_uva

    def copy_to_uva(n=None):
        if not self._num_blocks_dirty:
            return self.num_blocks.gpu
        self._num_blocks_dirty = False
        return orig_num_blocks_copy(n)

    self.num_blocks.copy_to_uva = copy_to_uva


def _append_block_ids(self, req_index, new_block_ids, overwrite):
    _orig_append_block_ids(self, req_index, new_block_ids, overwrite)
    self._num_blocks_dirty = True


def _sampler_add_request(self, req_idx, sampling_params):
    _orig_sampler_add_request(self, req_idx, sampling_params)
    self._has_staged_writes = True


def _sampler_apply_staged_writes(self):
    if not getattr(self, "_has_staged_writes", False):
        return
    self._has_staged_writes = False
    _orig_sampler_apply_staged_writes(self)


BlockTables.__init__ = _block_tables_init
BlockTables.append_block_ids = _append_block_ids
Sampler.add_request = _sampler_add_request
Sampler.apply_staged_writes = _sampler_apply_staged_writes
