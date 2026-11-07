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
