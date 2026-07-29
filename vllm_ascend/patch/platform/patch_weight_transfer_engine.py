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
# Patch target: vllm.distributed.weight_transfer.factory.WeightTransferEngineFactory
#
# Replace the "nccl" and "ipc" factory entries with Ascend equivalents so that
# --weight-transfer-config '{"backend": "nccl"}' loads HCCLWeightTransferEngine
# and '{"backend": "ipc"}' loads NPUIPCWeightTransferEngine instead of the
# (unavailable) NCCL / CUDA IPC engines on Ascend NPU.
#
# Why this approach (factory swap):
#   WeightTransferConfig.backend accepts custom strings, so users can pass
#   "hccl" / "npu_ipc" directly. We still replace the upstream-compatible
#   "nccl" / "ipc" aliases on Ascend so existing weight transfer examples and
#   configs resolve to HCCL / NPU IPC instead of CUDA-specific engines.
#
# Timing — guaranteed to run before first factory usage:
#
#   vllm serve main()
#     line 24: from vllm.entrypoints.utils import ...
#       → vllm.platforms.__getattr__("current_platform")
#       → resolve_current_platform_cls_qualname()
#       → vllm_ascend:register() → NPUPlatform()
#       → NPUPlatform.pre_register_and_update()
#       → adapt_patch(is_global_patch=True)
#       → imports vllm_ascend.patch.platform
#       → THIS PATCH RUNS  ← "nccl" now points to HCCLWeightTransferEngine
#     ...
#     lines 82-86: subparser_init() → make_arg_parser()
#     line 87: parse_args() → validates backend="nccl" via Literal (passes)
#     ...
#     later: worker init → WeightTransferEngineFactory.create_engine(config)
#       → config.backend == "nccl" → factory loads HCCLWeightTransferEngine
#
# Future Plan:
#   Remove this patch when upstream vLLM provides a stable extension point for
#   out-of-tree backend aliases.

from vllm_ascend.distributed.weight_transfer.registry import register_ascend_weight_transfer_engines


register_ascend_weight_transfer_engines()
