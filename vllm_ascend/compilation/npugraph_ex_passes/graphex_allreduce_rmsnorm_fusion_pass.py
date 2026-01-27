# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch
import torchair
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.compilation.npugraph_ex_passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# computation-communication tiling block is 512
ALLREDUCE_NORM_FUSE_THREHOLD = 512


class GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern:
    """
    recognizing the Matmul + AllReduce + AddRMSNorm computation pattern
    AllReduce is optimized in the fusion operator to a two-stage communication of ReduceScatter+AllGather
    """

    def __init__(self, vllm_config, eps=1e-6):
        self.vllm_config = vllm_config
        self.eps = eps
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        batch_size, seq_len = 2, 4
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def register(self):
        def pattern(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_, residual, rms_norm_weight, None)
            out0 = output[0]
            out1 = output[2]

            return out0, out1

        def replacement(x, weight, residual, rms_norm_weight):
            out0, out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                x,
                weight,
                residual,
                rms_norm_weight,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,
                False,
            )
            return out0, out1

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXLastLayerMatmulAllReduceAddRMSNormPattern:
    def __init__(self, vllm_config, eps=1e-6):
        self.vllm_config = vllm_config
        self.eps = eps
        device_group = get_tp_group().device_group
        backend = device_group._get_backend(torch.device("npu"))
        self.local_rank = torch.distributed.get_rank(group=device_group)
        self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self):
        batch_size, seq_len = 2, 4
        hidden_size = 4096
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        weight = torch.randn(hidden_size, hidden_size, device="npu")
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu")
        rms_norm_weight = torch.randn(hidden_size, device="npu")
        return [x, weight, residual, rms_norm_weight]

    def register(self):
        def pattern(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_, residual, rms_norm_weight, None)

            return output[0]

        def replacement(x, weight, residual, rms_norm_weight):
            out0, _ = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                x,
                weight,
                residual,
                rms_norm_weight,
                self.tp_group_name,
                self.tp_size,
                self.local_rank,
                self.eps,
                True,
                False,
            )
            return out0

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXMatmulAllReduceAddRMSNormPass:
    def __init__(self, vllm_config: VllmConfig):
        GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern(vllm_config).register()
        GraphEXLastLayerMatmulAllReduceAddRMSNormPattern(vllm_config).register()

    def __call__(self, graph: torch.fx.Graph):
        pass

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        applicable = compile_range.start > ALLREDUCE_NORM_FUSE_THREHOLD
        return applicable
