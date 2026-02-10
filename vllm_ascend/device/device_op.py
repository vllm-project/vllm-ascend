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

import torch_npu

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache, slot_indices=slot_mapping
        )
    @staticmethod
    def quant_apply_mlp(**kwargs):
        from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp as _impl
        return _impl(**kwargs)


class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key, value=value.contiguous(), key_cache=key_cache, value_cache=value_cache, slot_mapping=slot_mapping
        )
    @staticmethod
    def quant_apply_mlp(**kwargs):
        from vllm_ascend.ops.fused_moe.moe_mlp import quant_apply_mlp_A5 as _impl
        return _impl(**kwargs)

def get_device_adaptor():
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


DeviceOperator: type["BaseDeviceAdaptor"] | None = get_device_adaptor()
