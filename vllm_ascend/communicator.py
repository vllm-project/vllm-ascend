import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.base_communicator import (
    CommunicatorBase)


class NPUCommunicator(CommunicatorBase):

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, group=self.group)
        return x
