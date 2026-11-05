#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.utils import collect_pp_group_owners


def test_collect_pp_group_owners_without_pipeline_parallelism():
    assert collect_pp_group_owners(None, 1, 2) == ((0,), (0,))


def test_collect_pp_group_owners_gathers_stage_local_manifests():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.self_attn"]),
            SimpleNamespace(layer_names=[]),
        ]
    )

    def gather(gathered, local_presence, group):
        assert local_presence == (True, False)
        assert group == "cpu-group"
        gathered[:] = [(True, False), (False, True)]

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.utils.torch.distributed.all_gather_object",
        side_effect=gather,
    ):
        assert collect_pp_group_owners(config, 2, 2, "cpu-group") == ((0,), (1,))


def test_collect_pp_group_owners_falls_back_for_inconsistent_manifests():
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["layer"]), SimpleNamespace(layer_names=[])])

    def gather(gathered, local_presence, group):
        gathered[:] = [(True, False), (True,)]

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.utils.torch.distributed.all_gather_object",
        side_effect=gather,
    ):
        assert collect_pp_group_owners(config, 2, 2, "cpu-group") is None
