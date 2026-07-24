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

"""Monkey-patch vLLM's eager_break_during_capture for Ascend breakable ACL graph.

``sparse_attn_indexer`` does ``from vllm.compilation.breakable_cudagraph import
eager_break_during_capture`` — this creates a local binding at import time.
We replace the function in the original module BEFORE ``sparse_attn_indexer`` is
loaded, so its ``from ... import`` resolves to the Ascend variant that uses
``BreakableACLGraphCapture`` / ``torch.npu.NPUGraph`` instead of
``BreakableCUDAGraphCapture`` / ``torch.cuda.graph``.
"""

from vllm.compilation import breakable_cudagraph

from vllm_ascend.compilation.breakable_aclgraph import eager_break_during_capture

breakable_cudagraph.eager_break_during_capture = eager_break_during_capture
