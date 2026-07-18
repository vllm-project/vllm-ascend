#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
from torch._functorch import config as functorch_config

from vllm_ascend.compilation.compiler_interface import _disable_incompatible_aot_autograd_cache


def test_disable_incompatible_aot_autograd_cache_is_scoped():
    outer_cache_config = {"bundled_autograd_cache": True}
    if hasattr(functorch_config, "force_autograd_cache"):
        outer_cache_config["force_autograd_cache"] = True

    with functorch_config.patch(outer_cache_config):
        with _disable_incompatible_aot_autograd_cache():
            assert functorch_config.bundled_autograd_cache is False
            if hasattr(functorch_config, "force_autograd_cache"):
                assert functorch_config.force_autograd_cache is False

        assert functorch_config.bundled_autograd_cache is True
        if hasattr(functorch_config, "force_autograd_cache"):
            assert functorch_config.force_autograd_cache is True
