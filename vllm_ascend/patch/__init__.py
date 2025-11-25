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

# ----------------------------------------------------------------------------------
# This module manage the patch for vllm. There are two folders in this module:
# - platform: contains the patches applied before worker starts. It's called by
#             `vllm_ascend.utils.adapt_patch(is_global_patch=True)` in
#             `vllm_ascend.platform.NPUPlatform.pre_register_and_update()` function.
# - worker: contains the patches applied when worker starts. It's called by
#           `vllm_ascend.utils.adapt_patch(is_global_patch=False)` in
#           each worker's `__init__` function.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# * Platform Patch:
# =================
# ** File: platform/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.ParallelConfig.get_next_dp_init_port`
#    Why:
#       vllm doesn't support get port from environment.
#    How：
#       Add the logic to get port from environment.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support get port from environment.
#    Future Plan:
#       Remove those patch when vllm merged them
#   2. `torch.distributed.all_reduce`, `torch.distributed.broadcast`
#    Why:
#       tensor alignment for 310p
#    How：
#       rewrite all_reduce and broadcast in torch.distributed
#    Related PR (if no, explain why):
#       No, not ready yet.
#    Future Plan:
#       Find a better way to support tensor alignment for 310p without this patch.
#
# ** File: worker/patch_multimodal_merge.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.utils._merge_multimodal_embeddings`
#    Why:
#       '_merge_multimodal_embeddings' func of vllm is incompatible with Ascend.
#    How：
#       Replace with CPU operation that can be executed asynchronously.
#    Related PR (if no, explain why):
#       This is a bug by Ascend only. It can' be fixed in vLLM.
#    Future Plan:
#       Identify this pattern in torch-npu and remove this patch.
#
# * Worker Patch:
# ===============
# ** File: worker/patch_minicpm.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.minicpm.MiniCPMAttention.forward`
#    Why:
#       The forward func of MiniCPMAttention in vllm do a datatype convert
#       (original datatype --> float32) to ensure the precision on cuda.
#       However float32 is not supported in cann rope op, thus we keep this patch
#    How：
#       Removed the dtype convert operations in forward
#    Related PR (if no, explain why):
#       NO, only for npu due to rope op.
#    Future Plan:
#       Keep this patch in vllm-ascend.
#
# ** File: worker/patch_distributed.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.parallel_state.GroupCoordinator`
#   (1) __init__()
#    Why:
#       The original GroupCoordinator initialization lacks pg_options to generate new
#       process group with customized options.
#    How:
#       Inject HCCL options during process group initialization.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support a dictionary as input while initializing distributed
#       environment (e.g., Dict[str, torch.distributed.ProcessGroupHCCL.Options])
#       https://github.com/vllm-project/vllm/pull/25417
#    Future Plan:
#       Remove this patch when vllm merges this PR.
#   (2) all_to_all()
#    Why:
#       vllm doesn't support all_to_all for GroupCoordinator.
#    How：
#       Add all_to_all implementation for GroupCoordinator.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support all_to_all for GroupCoordinator.
#    Future Plan:
#       Remove this patch when vllm merged them.
#
# ** File: worker/patch_roberta.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.roberta.RobertaEmbedding.forward`
#    Why:
#       shift operation in `_encode_token_type_ids` and `_decode_token_type_ids` cannot run in ascend aclgraph mode
#    How：
#       Replace shift operation with multiplication and division.
#    Related PR (if no, explain why):
#       No, this need CANN add an aclnn shift operation
#    Future Plan:
#       Revert this when CANN support shift aclnn operation
#   2. `vllm.model_executor.models.roberta.RobertaForSequenceClassification.forward `
#    Why:
#       shift operation in `_encode_token_type_ids` and `_decode_token_type_ids` cannot run in ascend aclgraph mode
#    How：
#       Replace shift operation with multiplication and division.
#    Related PR (if no, explain why):
#       No, this need CANN add an aclnn shift operation
#    Future Plan:
#       Revert this when CANN support shift aclnn operation
#
# ** File: worker/patch_deepseek_mtp.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer.__init__`
#    Why:
#       '__init__' func of DeepSeekMultiTokenPredictorLayer didn't pass prefix to SharedHead.
#    How：
#       Replace with a new __init__.
#       Use a new SharedHead which passes prefix to ParallelLMHead.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/25805
#    Future Plan:
#       Remove this patch when adapted vllm version contains the above PR.
#
