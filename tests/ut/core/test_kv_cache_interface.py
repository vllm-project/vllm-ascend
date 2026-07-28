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
# This file is a part of the vllm-ascend project.

import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


def test_ascend_mla_page_size_honors_hybrid_cache_padding():
    spec = AscendMLAAttentionSpec(
        block_size=768,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=912384,
    )

    assert spec.real_page_size_bytes == 884736
    assert spec.page_size_bytes == 912384

    merged = AscendMLAAttentionSpec.merge([spec, spec])
    assert merged.real_page_size_bytes == 884736
    assert merged.page_size_padded == 912384
    assert merged.page_size_bytes == 912384
