#
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
#

import torch
import torch_npu
from vllm.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group)
from vllm.logger import logger


class P2PLoad:

    def __init__(
        self,
        world_name: str,
        source_ip: str,
        source_port: int,
    ):
        self.world_name = world_name
        self.source_ip = source_ip
        self.source_port = source_port

    def load(self, model):
        model_device = next(model.parameters()).device
        logger.info(
            f"Start init_process_group, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
        )
        try:
            receiver_pg = stateless_init_torch_distributed_process_group(
                host=self.world_name.split(":")[0],
                port=self.source_port,
                rank=0,
                world_size=2,
                backend='hccl',
            )
            logger.info(
                f"Finish init_process_group, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
            )

            logger.info(
                f"Start recv, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
            )
            logger.info(f"Model device: {model_device}")

            trans_stream = torch_npu.npu.Stream()
            with torch_npu.npu.stream(trans_stream):
                handlers = []
                for name, param in model.named_parameters():
                    if len(param.shape) == 0:
                        continue
                    handlers.append(receiver_pg.recv([param], 1, 0))
                for h in handlers:
                    h.wait()
                torch.distributed.barrier(group=receiver_pg,
                                          device_ids=[model_device.index])

            torch_npu.npu.synchronize(trans_stream)

            logger.info(
                f"Finish recv, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
            )
        except Exception as e:
            logger.error("Failed to recv model: {}".format(e))
            if 'receiver_pg' in locals():
                stateless_destroy_torch_distributed_process_group(receiver_pg)
            return None

        stateless_destroy_torch_distributed_process_group(receiver_pg)
        return model


class P2PSend:

    def __init__(self, listen_ip: str, listen_port: int, comm_name: str):
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.comm_name = comm_name

    def send(self, model, int8_params: dict):
        model_device = next(model.parameters()).device
        torch.npu.set_device(model_device)
        logger.info(
            f"Start init_process_group, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
        )
        sender_pg = stateless_init_torch_distributed_process_group(
            host=self.comm_name.split(":")[0],
            port=self.listen_port,
            rank=1,
            world_size=2,
            backend='hccl',
        )
        logger.info(
            f"Finish init_process_group, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
        )
        logger.info(
            f"Start send, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
        )
        logger.info(f"Model device: {model_device}")

        trans_stream = torch_npu.npu.Stream()
        with torch_npu.npu.stream(trans_stream):
            handlers = []
            for name, param in model.named_parameters():
                if name in int8_params:
                    handlers.append(
                        sender_pg.send([int8_params[name].to(model_device)], 0,
                                       0))
                else:
                    handlers.append(sender_pg.send([param.contiguous()], 0, 0))
            for h in handlers:
                h.wait()
            torch.distributed.barrier(group=sender_pg,
                                      device_ids=[model_device.index])
        torch_npu.npu.synchronize(trans_stream)
        logger.info(
            f"Finish send, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
        )
        stateless_destroy_torch_distributed_process_group(sender_pg)
