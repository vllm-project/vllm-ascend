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
# This patch fixes an upstream vLLM bug where `WorkerWrapperBase.adjust_rank()`
# updates `rpc_rank` but not `global_rank`, causing workers to select the wrong
# `kv_cache_config` during initialization when using Ray executor with
# pipeline parallelism (PP >= 3).
#
# Upstream issue: https://github.com/vllm-project/vllm/issues/30128
# Upstream PR: https://github.com/vllm-project/vllm/pull/33700

from vllm.v1.worker.worker_base import WorkerWrapperBase


def _patched_adjust_rank(self, rank_mapping: dict[int, int]) -> None:
    """
    Adjust both rpc_rank and global_rank based on the given mapping.

    This fixes a bug in the upstream vLLM where only rpc_rank was updated
    but global_rank was left stale. When the Ray executor remaps worker
    ranks during pipeline parallel initialization, global_rank must also
    be updated so that `initialize_from_config` can correctly select the
    per-worker KVCacheConfig from the global list using self.global_rank.

    Without this fix, workers receive the wrong PP stage's kv_cache_config,
    causing a KeyError in get_layers_from_vllm_config because the layer
    names in the config don't exist in the worker's static_forward_context.
    """
    if self.rpc_rank in rank_mapping:
        self.global_rank = self.rpc_rank = rank_mapping[self.rpc_rank]


WorkerWrapperBase.adjust_rank = _patched_adjust_rank
