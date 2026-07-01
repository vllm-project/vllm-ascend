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

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, cast

import torch
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import GroupCoordinator, _get_unique_name, _register_group

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.patch.worker._hccl_pg_registry import HcclPgRegistry, make_hccl_pg_key
from vllm_ascend.utils import create_hccl_pg_options

_HCCL_PG_REGISTRY = HcclPgRegistry()
logger = logging.getLogger(__name__)


def _normalize_backend(backend: str | Backend) -> str:
    return str(backend)


def _resolve_reuse_domain(group_name: str) -> str:
    group_base_name = group_name.split(":")[0]
    if "eplb" in group_base_name or group_base_name == "mc2":
        return group_base_name
    return "shared"


def _create_device_group(
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
):
    return torch.distributed.new_group(
        ranks,
        backend=backend,
        pg_options=hccl_pg_options,
    )


def _acquire_hccl_group(
    *,
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
    reuse_domain: str,
):
    # Coordinator construction must remain process-serial and globally ordered:
    # new_group is collective, and the registry only deduplicates equivalent
    # HCCL groups within that ordering contract. It is not a concurrent PG factory.
    hccl_key = make_hccl_pg_key(ranks, backend, hccl_pg_options, reuse_domain)
    device_group = _HCCL_PG_REGISTRY.acquire(
        ranks=ranks,
        backend=backend,
        pg_options=hccl_pg_options,
        reuse_domain=reuse_domain,
        create_fn=lambda: _create_device_group(ranks, backend, hccl_pg_options),
    )
    return device_group, hccl_key


def _wrap_destroy_distributed_environment(destroy_fn):
    if getattr(cast(Any, destroy_fn), "_hccl_registry_clearing_wrapped", False) is True:
        return destroy_fn

    @wraps(destroy_fn)
    def wrapped(*args, **kwargs):
        try:
            return destroy_fn(*args, **kwargs)
        finally:
            _HCCL_PG_REGISTRY.clear()

    cast(Any, wrapped)._hccl_registry_clearing_wrapped = True
    return wrapped


def _patch_destroy_distributed_environment():
    destroy_fn = _wrap_destroy_distributed_environment(vllm.distributed.parallel_state.destroy_distributed_environment)
    vllm.distributed.parallel_state.destroy_distributed_environment = destroy_fn
    vllm.distributed.destroy_distributed_environment = destroy_fn


class GroupCoordinatorPatch(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.backend = _normalize_backend(torch_distributed_backend)
        self._acquired_hccl_keys = []
        self._unshared_hccl_groups = []
        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        self.mq_broadcaster = None
        self.cpu_group = None
        self.device_group = None
        self.device = None
        self.use_custom_op_call = True
        self.use_cpu_custom_send_recv = False

        reuse_domain = _resolve_reuse_domain(group_name)

        try:
            for ranks in group_ranks:
                hccl_pg_options = create_hccl_pg_options(group_name)
                device_group, hccl_key = _acquire_hccl_group(
                    ranks=ranks,
                    backend=self.backend,
                    hccl_pg_options=hccl_pg_options,
                    reuse_domain=reuse_domain,
                )
                if hccl_key is not None:
                    self._acquired_hccl_keys.append(hccl_key)
                elif self.backend == "hccl" and self.rank in ranks:
                    self._unshared_hccl_groups.append(device_group)

                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group

            assert self.cpu_group is not None
            assert self.device_group is not None

            self.device = torch.npu.current_device()
            if use_device_communicator and self.world_size > 1:
                self.device_communicator = NPUCommunicator(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    unique_name=self.unique_name,
                )

            from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

            if use_message_queue_broadcaster and self.world_size > 1:
                self.mq_broadcaster = MessageQueue.create_from_process_group(
                    self.cpu_group,
                    1 << 22,
                    6,
                )
        except Exception:
            try:
                self.destroy()
            except Exception:
                logger.exception("Failed to clean up partially initialized GroupCoordinatorPatch")
            raise

    def destroy(self):
        cpu_group = getattr(self, "cpu_group", None)
        if cpu_group is not None:
            torch.distributed.destroy_process_group(cpu_group)
        if hasattr(self, "cpu_group"):
            del self.cpu_group

        if hasattr(self, "_acquired_hccl_keys"):
            for hccl_key in reversed(self._acquired_hccl_keys):
                _HCCL_PG_REGISTRY.release(hccl_key)
            self._acquired_hccl_keys = []

        if hasattr(self, "_unshared_hccl_groups"):
            for device_group in reversed(self._unshared_hccl_groups):
                torch.distributed.destroy_process_group(device_group)
            self._unshared_hccl_groups = []

        device_group = getattr(self, "device_group", None)
        if device_group is not None and self.backend != "hccl":
            torch.distributed.destroy_process_group(device_group)
        if hasattr(self, "device_group"):
            del self.device_group

        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            device_communicator.destroy()
            self.device_communicator = None

        if getattr(self, "mq_broadcaster", None) is not None:
            self.mq_broadcaster = None

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
_patch_destroy_distributed_environment()
