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

from collections import deque
import pickle
from functools import wraps
from typing import Any, cast

import torch
import vllm
from torch import Tensor
from torch.distributed import Backend, ProcessGroup, Work
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _get_unique_name,
    _register_group,
    _split_tensor_dict,
)
from vllm.logger import logger

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator
from vllm_ascend.patch.worker._hccl_pg_registry import HcclPgKey, HcclPgRegistry, make_hccl_pg_key
from vllm_ascend.utils import create_hccl_pg_options
from vllm_ascend.distributed.parallel_state import (
    get_virtual_pipeline_parallel_rank,
    get_virtual_pipeline_parallel_size,
    get_vpp_runtime_active,
)
from vllm_ascend.distributed.vpp_utils import is_vpp_last_stage as _is_vpp_last_stage

_HCCL_PG_REGISTRY = HcclPgRegistry()


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
        # One entry per async send_tensor_dict call: (isend handles, retained
        # source tensors). The tensor refs keep the source memory alive while
        # the non-blocking sends are in flight, so it is neither freed nor
        # overwritten before completion (cf. async-send data-corruption fix).
        self._async_send_buff: deque[tuple[list[Work], list[Tensor]]] = deque()
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.group_ranks = group_ranks
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.backend = _normalize_backend(torch_distributed_backend)
        self._acquired_hccl_keys: list[HcclPgKey] = []
        self._unshared_hccl_groups: list[object] = []
        self.use_device_communicator = use_device_communicator
        self.device_communicator: NPUCommunicator | None = None
        self.mq_broadcaster = None
        self.cpu_group = None
        self.device_group = None
        self.device = None
        self.use_custom_op_call = True

        self.vpp_size = get_virtual_pipeline_parallel_size()
        self.vpp_stage = get_virtual_pipeline_parallel_rank()

        self.torch_distributed_backend = torch_distributed_backend
        self.use_message_queue_broadcaster = use_message_queue_broadcaster
        self.use_cpu_custom_send_recv = False
        self.group_name = group_name
        self.group_ranks = group_ranks

        try:
            self._init_device_groups(create_cpu_group=True)
            assert self.cpu_group is not None
            assert self.device_group is not None

            self._init_device_communicator()

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

    def _init_device_groups(self, create_cpu_group: bool) -> None:
        reuse_domain = _resolve_reuse_domain(self.group_name)
        self_device_group = None
        for ranks in self.group_ranks:
            hccl_pg_options = create_hccl_pg_options(self.group_name)
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

            cpu_group = torch.distributed.new_group(ranks, backend="gloo") if create_cpu_group else None
            if self.rank in ranks:
                if create_cpu_group:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.cpu_group = cpu_group
                self_device_group = device_group

        if self_device_group is not None:
            self.device_group = self_device_group

    def _init_device_communicator(self) -> None:
        self.device = torch.npu.current_device()
        if self.use_device_communicator and self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

    def _release_hccl_resources(self) -> bool:
        destroyed = False
        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            device_communicator.destroy()
            self.device_communicator = None
            destroyed = True

        if hasattr(self, "_acquired_hccl_keys"):
            for hccl_key in reversed(self._acquired_hccl_keys):
                _HCCL_PG_REGISTRY.release(hccl_key)
            self._acquired_hccl_keys = []
            destroyed = True

        if hasattr(self, "_unshared_hccl_groups"):
            for device_group in reversed(self._unshared_hccl_groups):
                torch.distributed.destroy_process_group(device_group)
            self._unshared_hccl_groups = []
            destroyed = True

        return destroyed

    def destroy(self):
        if getattr(self, "mq_broadcaster", None) is not None:
            self.mq_broadcaster = None

        self._release_hccl_resources()

        device_group = getattr(self, "device_group", None)
        if device_group is not None and self.backend != "hccl":
            torch.distributed.destroy_process_group(device_group)
        if hasattr(self, "device_group"):
            del self.device_group

        cpu_group = getattr(self, "cpu_group", None)
        if cpu_group is not None:
            torch.distributed.destroy_process_group(cpu_group)
        if hasattr(self, "cpu_group"):
            del self.cpu_group

    def destroy_hccl(self) -> bool:
        """Release the HCCL process group."""
        destroyed = self._release_hccl_resources()

        if hasattr(self, "device_group"):
            self.device_group = None
        return destroyed

    def restore_hccl(self) -> bool:
        """Recreate the HCCL process group in place after sleep mode."""
        if self.device_group is not None:
            return False

        self._init_device_groups(create_cpu_group=False)
        assert self.device_group is not None
        self._init_device_communicator()
        return True

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

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group.

        Under VPP this is *phase-dependent*:

        * Construction (``get_vpp_runtime_active()`` is False, i.e. before
          the VPP schedule loop starts): return the **static** topology
          answer -- the first virtual stage always starts at pp_rank 0 --
          so model ``__init__`` places ``embed_tokens`` on the rank that
          owns the first stage.
        * Runtime (after the loop starts): return the **dynamic** answer
          for the current virtual stage, used by ``forward`` to decide
          whether to embed the input ids.
        """
        if get_virtual_pipeline_parallel_size() <= 1:
            return self.rank == self.first_rank
        if not get_vpp_runtime_active():
            return self.rank_in_group == 0
        return self.rank_in_group == 0 and get_virtual_pipeline_parallel_rank() == 0
    
    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group.

        Under VPP this is *phase-dependent*:

        * Construction (``get_vpp_runtime_active()`` is False, i.e. before
          the VPP schedule loop starts): return the **static** fold-back
          topology answer -- which physical rank owns the very last
          virtual stage -- so model ``__init__`` places ``norm`` /
          ``lm_head`` on the correct rank. This is ``is_vpp_last_stage``
          evaluated at ``vp_stage = vp_size - 1`` and does **not** depend
          on the (still-zero) live virtual rank.
        * Runtime (after the loop starts): return the **dynamic** answer
          for the current virtual stage, used by ``forward`` to decide
          whether to apply norm / emit logits.
        """
        vp_size = get_virtual_pipeline_parallel_size()
        if vp_size <= 1:
            return self.rank == self.last_rank
        pp_rank = self.rank_in_group
        pp_size = self.world_size
        if not get_vpp_runtime_active():
            return self.is_vpp_last_stage(
                pp_rank=pp_rank, pp_size=pp_size,
                vp_stage=vp_size - 1, vp_size=vp_size)
        vp_stage = get_virtual_pipeline_parallel_rank()
        return self.is_vpp_last_stage(
            pp_rank=pp_rank, pp_size=pp_size,
            vp_stage=vp_stage, vp_size=vp_size)
    
    def is_vpp_last_stage(
        self,
        pp_rank: int,
        pp_size: int,
        vp_stage: int,
        vp_size: int,
    ) -> bool:
        """Delegate to the shared implementation in ``vpp_utils`` so the
        fold-back semantics are defined exactly once.
        """
        return _is_vpp_last_stage(pp_rank, pp_size, vp_stage, vp_size)

    def _release_completed_send_buff(self) -> None:
        """Lazily drop async-send batches whose hccl sends have completed.

        gloo ``Work.is_completed()`` is unreliable on this backend, so we gate
        on the hccl handles only (``handles[2:]``, reliable) and ``wait()`` the
        gloo metadata handles (``handles[:2]``) as a backstop before dropping.
        The waited entry is old (its hccl already completed) so the wait is
        ~instant and does NOT reintroduce the fold-point deadlock (that
        deadlock comes from blocking on a *current* send; here we block on an
        already-completed old one).

        Metadata-only entries (``handles[2:]`` empty, e.g. empty
        ``IntermediateTensors``) have no hccl gate and used to leak forever
        because ``not hccl_handles`` short-circuited the loop. Drop them
        best-effort: pop without blocking ``wait()`` on the gloo metadata
        (which would reintroduce the fold-point deadlock risk). The retained
        ``size_tensor`` / ``object_tensor`` go out of scope on pop, releasing
        the only large memory; the small gloo ``Work`` handles may dangle but
        are bounded.
        """
        while self._async_send_buff:
            handles = self._async_send_buff[0][0]
            # handles[:2] = gloo metadata；handles[2:] = hccl tensor（is_completed 可靠）
            hccl_handles = handles[2:]
            if not hccl_handles:
                # Metadata-only send: no reliable completion signal, just drop.
                self._async_send_buff.popleft()
                continue
            if not all(h.is_completed() for h in hccl_handles):
                break   # 队首 hccl 尚未完成 → FIFO 后续更未完成，停止
            popped = self._async_send_buff.popleft()
            for h in popped[0][:2]:   # gloo metadata wait 兜底，确保已发出
                h.wait()

    def _wait_send_buff(self) -> None:
        """Block until every in-flight async send has completed."""
        while self._async_send_buff:
            handles, _ = self._async_send_buff.popleft()
            for handle in handles:
                handle.wait()

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
        is_async: bool = False,
    ) -> dict[str, Any] | None:
        """Send a tensor dict, optionally fully non-blocking (``is_async=True``).

        ``isend_tensor_dict`` is overridden so the metadata is also posted with
        ``isend`` (upstream sends it via a blocking ``send_object``). The VPP
        stage schedule needs every point-to-point op non-blocking, otherwise a
        fold-point send whose matching recv lives in a later iteration deadlocks
        both ranks inside the metadata send.
        """
        if not is_async:
            return super().send_tensor_dict(
                tensor_dict,
                dst=dst,
                all_gather_group=all_gather_group,
                all_gather_tensors=all_gather_tensors,
            )
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        return None

    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list:
        """Non-blocking tensor-dict send: metadata and tensors both use isend.

        Retains references to every source/metadata tensor in
        ``_async_send_buff`` until the sends complete, so the memory is neither
        freed nor overwritten while in flight.
        """
        if self.world_size <= 1:
            return []
        # Non-VPP (vp_size <= 1): fall back to upstream blocking isend so it
        # pairs with the upstream irecv_tensor_dict on the peer rank. The async
        # path below (non-blocking metadata isend + FIFO release) is only
        # correct under the VPP stage schedule; under regular PP it deadlocks
        # against the sync recv on the peer.
        if get_virtual_pipeline_parallel_size() <= 1:
            return super().isend_tensor_dict(
                tensor_dict,
                dst=dst,
                all_gather_group=all_gather_group,
                all_gather_tensors=all_gather_tensors,
            )
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        self._release_completed_send_buff()
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )
        group = self.device_group
        metadata_group = self.cpu_group

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # Non-blocking metadata send (the deadlock fix: upstream blocks here on
        # send_object while the matching recv is posted in a later iteration).
        handles, retained = self._isend_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue
            if self._should_use_all_gather(
                key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            if handle is not None:
                handles.append(handle)
            retained.append(tensor)

        if handles:
            self._async_send_buff.append((handles, retained))
        return handles

    def _isend_object(
        self, obj: Any, dst: int
    ) -> tuple[list, list[Tensor]]:
        """Non-blocking send_object: isend the size + pickled object, retain refs."""
        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )
        retained = [size_tensor, object_tensor]
        handles: list = []
        for tensor in retained:
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=self.cpu_group
            )
            if handle is not None:
                handles.append(handle)
        return handles, retained


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
_patch_destroy_distributed_environment()
