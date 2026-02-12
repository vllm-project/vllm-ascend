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
from vllm.logger import logger

from .netloader_pg import (destroy_stateless_process_group,
                           stateless_init_process_group)


class P2PLoad:
    """
    Class for receiving model parameters in a distributed manner using HCCL backend.
    """

    def __init__(
        self,
        world_name: str,
        source_ip: str,
        source_port: int,
    ):
        """
        Initializes the P2PLoad instance.

        Parameters:
        - world_name: The name of the distributed group.
        - source_ip: The IP address of the source node.
        - source_port: The port number for the source node.
        """
        self.world_name = world_name
        self.source_ip = source_ip
        self.source_port = source_port

    def load(self, model):
        """
        Loads the model parameters using HCCL backend.

        Parameters:
        - model: The model whose parameters are to be loaded.

        Returns:
        - The model if loading is successful, otherwise None.
        """
        model_device = next(model.parameters()).device
        logger.info(
            f"Start init_process_group, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
        )
        receiver_pg = None
        loaded_model = None
        try:
            receiver_pg = stateless_init_process_group(
                host=self.world_name.split(":")[0],
                port=self.source_port,
                rank=0,
                world_size=2,
                group_name='netloader',
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
                for name, param in model.named_parameters():
                    if len(param.shape) == 0:
                        continue
                    receiver_pg.recv([param], 1, 0).wait()
                torch.distributed.barrier(group=receiver_pg,
                                          device_ids=[model_device.index])

            torch_npu.npu.synchronize(trans_stream)

            logger.info(
                f"Finish recv, name: {self.world_name}, addr: {self.source_ip}:{self.source_port}"
            )
            loaded_model = model
        except Exception as e:
            logger.error("Failed to recv model: {}".format(e))
        finally:
            if receiver_pg:
                destroy_stateless_process_group(receiver_pg)
        return loaded_model


class P2PSend:
    """
    Class for sending model parameters in a distributed manner using HCCL backend.
    """

    def __init__(self, listen_ip: str, listen_port: int, comm_name: str):
        """
        Initializes the P2PSend instance.

        Parameters:
        - listen_ip: The IP address to listen on.
        - listen_port: The port number to listen on.
        - comm_name: The name of the communication group.
        """
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.comm_name = comm_name

    def send(self, model, int8_params: dict):
        """
        Sends the model parameters using HCCL backend.

        Parameters:
        - model: The model whose parameters are to be sent.
        - int8_params: Dictionary of parameters that are in int8 format.
        """
        model_device = next(model.parameters()).device
        torch.npu.set_device(model_device)
        logger.info(
            f"Start init_process_group, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
        )
        sender_pg = None
        try:
            sender_pg = stateless_init_process_group(
                host=self.comm_name.split(":")[0],
                port=self.listen_port,
                rank=1,
                world_size=2,
                group_name='netloader',
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
                for name, param in model.named_parameters():
                    if "aclnn_input_scale" in name:
                        continue
                    if name in int8_params:
                        sender_pg.send([int8_params[name].to(model_device)], 0,
                                       0).wait()
                    else:
                        sender_pg.send([param.contiguous()], 0, 0).wait()
                torch.distributed.barrier(group=sender_pg,
                                          device_ids=[model_device.index])
            torch_npu.npu.synchronize(trans_stream)
            logger.info(
                f"Finish send, name: {self.comm_name}, addr: {self.listen_ip}:{self.listen_port}"
            )
        finally:
            if sender_pg:
                destroy_stateless_process_group(sender_pg)