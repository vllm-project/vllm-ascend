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
import os

import pytest

from tests.conftest import VllmRunner
from vllm_ascend.ascend_config import (clear_ascend_config, get_ascend_config,
                                       init_ascend_config)


def _clean_up_ascend_config(func):

    def wrapper(*args, **kwargs):
        clear_ascend_config()
        func(*args, **kwargs)
        clear_ascend_config()

    return wrapper


@_clean_up_ascend_config
def test_run_without_ascend_config():
    with VllmRunner("facebook/opt-125m"):
        ascend_config = get_ascend_config()

        assert not ascend_config.torchair_graph_config.enabled
        assert not ascend_config.torchair_graph_config.use_cached_graph
        assert ascend_config.torchair_graph_config.graph_batch_sizes == []
        assert not ascend_config.torchair_graph_config.graph_batch_sizes_init
        assert not ascend_config.ascend_scheduler_config.enabled
        assert ascend_config.expert_tensor_parallel_size == 0


@_clean_up_ascend_config
def test_run_with_ascend_config():
    if os.getenv("VLLM_USE_V1") == "0":
        pytest.skip("graph only works on v1")

    input_additional_config_1 = {
        "torchair_graph_config": {
            # torchair graph only works with deepseek. The e2e test should be added
            # in multicard test with deepseek models.
            "enabled": False,
            "use_cached_graph": True,
            "graph_batch_sizes": [1, 2, 4, 8],
            "graph_batch_sizes_init": False,
            "enable_multistream_shared_expert": True,
        },
        "ascend_scheduler_config": {
            "enabled": True,
            "enable_chunked_prefill": True,
        },
        "expert_tensor_parallel_size": 1
    }

    # check passed with eager mode
    with VllmRunner("facebook/opt-125m",
                    enforce_eager=True,
                    additional_config=input_additional_config_1):
        ascend_config = get_ascend_config()

        assert not ascend_config.torchair_graph_config.enabled
        assert ascend_config.torchair_graph_config.use_cached_graph
        assert ascend_config.torchair_graph_config.graph_batch_sizes == [
            1, 2, 4, 8
        ]
        assert not ascend_config.torchair_graph_config.graph_batch_sizes_init
        assert ascend_config.torchair_graph_config.enable_multistream_shared_expert
        assert ascend_config.ascend_scheduler_config.enabled
        assert ascend_config.ascend_scheduler_config.enable_chunked_prefill
        assert ascend_config.expert_tensor_parallel_size == 1


@_clean_up_ascend_config
def test_ascend_config_init_error():
    # ascend_config should be initialized first
    with pytest.raises(RuntimeError):
        _ = get_ascend_config()


@_clean_up_ascend_config
def test_ascend_config_load_error():
    if os.getenv("VLLM_USE_V1") == "0":
        pytest.skip("graph only works on v1")
    # graph_batch_sizes should be list.
    with pytest.raises(TypeError):
        input_additional_config_fake_1 = {
            "torchair_graph_config": {
                "graph_batch_sizes": "fake_size",
            },
        }
        with VllmRunner("facebook/opt-125m",
                        additional_config=input_additional_config_fake_1):
            pass

    # graph_batch_sizes_init should not be True when graph_batch_sizes is not empty.
    with pytest.raises(ValueError):
        input_additional_config_fake_2 = {
            "torchair_graph_config": {
                "graph_batch_sizes": [1, 2, 4, 8],
                "graph_batch_sizes_init": True,
            },
        }
        with VllmRunner("facebook/opt-125m",
                        additional_config=input_additional_config_fake_2):
            pass

    # torchair graph only works with deepseek.
    with pytest.raises(NotImplementedError):
        input_additional_config_fake_2 = {
            "torchair_graph_config": {
                "enabled": True,
            },
        }
        with VllmRunner("facebook/opt-125m",
                        enforce_eager=False,
                        additional_config=input_additional_config_fake_2):
            pass

    # torchair graph should not be enabled with eager mode
    with pytest.raises(RuntimeError):
        input_additional_config_fake_3 = {
            "torchair_graph_config": {
                "enabled": True,
            },
        }
        with VllmRunner("facebook/opt-125m",
                        enforce_eager=True,
                        additional_config=input_additional_config_fake_3):
            pass


@_clean_up_ascend_config
def test_check_ascend_config_v0():
    if os.getenv("VLLM_USE_V1") == "1":
        pytest.skip("graph only works on v1, this is the test for v0")
    with pytest.raises(NotImplementedError):
        input_additional_config_fake_1 = {
            "torchair_graph_config": {
                "enabled": True,
            },
        }
        with VllmRunner("facebook/opt-125m",
                        additional_config=input_additional_config_fake_1):
            pass


@_clean_up_ascend_config
def test_ascend_config_refresh():
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    # set additional_config with none
    init_ascend_config(vllm_config)

    input_additional_config = {
        "torchair_graph_config": {
            "enabled": False,
            "use_cached_graph": True,
            "graph_batch_sizes": [1, 2, 4, 8],
            "graph_batch_sizes_init": False,
        },
        "refresh": True,
    }

    # refresh ascend config
    with VllmRunner("facebook/opt-125m",
                    additional_config=input_additional_config):
        ascend_config = get_ascend_config()

        assert not ascend_config.torchair_graph_config.enabled
        assert ascend_config.torchair_graph_config.use_cached_graph
        assert ascend_config.torchair_graph_config.graph_batch_sizes == [
            1, 2, 4, 8
        ]
        assert not ascend_config.torchair_graph_config.graph_batch_sizes_init
