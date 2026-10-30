#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Names under which Ascend offers online (load-time) quantization.

Kept deliberately free of imports: ``vllm_ascend.platform`` reads this while the
class body is being built, before the quantization package (and its torch_npu
kernels) can be imported.
"""

# ``--quantization`` values routed to vllm_ascend's online quantization config.
# ``online`` is the explicit form; ``int8_per_channel_weight_only`` is the
# upstream shorthand for "INT8 experts, dense layers untouched".
ASCEND_ONLINE_QUANT_METHODS: tuple[str, ...] = ("online", "int8_per_channel_weight_only")

__all__ = ["ASCEND_ONLINE_QUANT_METHODS"]
