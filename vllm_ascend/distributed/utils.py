import threading
from contextlib import contextmanager

import torch
import torch.distributed as dist
import vllm.distributed.stateless_coordinator as vllm_distributed_stateless_mod
from torch.distributed import ProcessGroup, Store
from torch.distributed.distributed_c10d import BackendConfig, _world
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group
from vllm.distributed.stateless_coordinator import (
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_fc3_quant_x_group


def fc3_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x
    x = get_fc3_quant_x_group().all_gather(x, 0)
    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = x[:-pad_size]
    else:
        # unpad
        num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
        result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
        dp_size = get_dp_group().world_size
        x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
            offset += num_tokens_dp
        x = result

    return x


def all_gather_async(
    input: torch.Tensor, group: GroupCoordinator, output: torch.Tensor | None = None, async_op: bool = True
):
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)


def stateless_init_pg_with_world_registration(**kwargs) -> ProcessGroup | tuple[ProcessGroup, Store]:
    if kwargs.get("return_store", False):
        pg, store = stateless_init_torch_distributed_process_group(**kwargs)
    else:
        pg = stateless_init_torch_distributed_process_group(**kwargs)

    backend = "hccl"
    prefix_store = pg.get_group_store()
    group_name = pg.group_name
    backend_config = BackendConfig(backend)

    # Register process group to PyTorch's global _world state
    # Required for: dist.P2POp, dist.batch_isend_irecv, and other ops that query _world.pg_map
    _world.pg_group_ranks[pg] = {i: i for i in range(pg.size())}
    _world.pg_map[pg] = (backend, prefix_store)
    _world.pg_names[pg] = group_name
    _world.pg_backend_config[pg] = str(backend_config)

    if "WORLD" in group_name:
        _world.default_pg = pg

    if kwargs.get("return_store", False):
        return pg, store
    else:
        return pg


def stateless_destroy_pg_with_world_cleanup(pg: ProcessGroup) -> None:
    stateless_destroy_torch_distributed_process_group(pg)

    # Remove related attributes from _world
    _world.pg_map.pop(pg, None)
    _world.pg_names.pop(pg, None)
    _world.pg_group_ranks.pop(pg, None)
    _world.pg_backend_config.pop(pg, None)


_PATCH_LOCK = threading.Lock()


@contextmanager
def use_stateless_pg_init_and_destroy_with_world():
    with _PATCH_LOCK:
        old_init_impl = stateless_init_torch_distributed_process_group
        old_destroy_impl = stateless_destroy_torch_distributed_process_group
        vllm_distributed_stateless_mod.stateless_init_torch_distributed_process_group = (
            stateless_init_pg_with_world_registration
        )
        vllm_distributed_stateless_mod.stateless_destroy_torch_distributed_process_group = (
            stateless_destroy_pg_with_world_cleanup
        )
        try:
            yield
        finally:
            vllm_distributed_stateless_mod.stateless_init_torch_distributed_process_group = old_init_impl
            vllm_distributed_stateless_mod.stateless_destroy_torch_distributed_process_group = old_destroy_impl
