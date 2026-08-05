# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


class ModelAclGraphManager310(ModelAclGraphManager):
    """ACL Graph manager for 310P direct-op capture and replay.

    Unlike mainline Ascend attention, 310P paged/splitfuse operators are
    captured directly and do not register graph-task handles that need to be
    updated before replay.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        model_runner,
        lora_capture_cases: list[int] | None = None,
    ) -> None:
        ModelCudaGraphManager.__init__(
            self,
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases=lora_capture_cases,
        )
        self.model_runner = model_runner
        self.capture_sizes = sorted({desc.num_tokens for descs in self._capture_descs.values() for desc in descs})

    def run_fullgraph(self, desc: BatchExecutionDescriptor):
        logger.info_once("run 310P full ACL Graph with num_tokens=%s", desc.num_tokens)
        return ModelCudaGraphManager.run_fullgraph(self, desc)
