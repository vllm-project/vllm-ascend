import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
import torch

from datetime import timedelta
from typing import Any, Optional, Union

import torch
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
    _get_default_group,
    _update_default_pg,
)
import pickle
from typing import Any, Callable, Optional, Union

class DefaultProcessGroupSwitcher:
    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)  

def creat_hccl_process_group(rank, world_size, attn_size, ffn_size):
    import torch
    torch.npu.set_device(rank)
    new_default_group = init_process_group(
        init_method='tcp://127.0.0.1:29500',
        backend='gloo', 
        rank=rank, 
        world_size=world_size, 
        group_name="new_hccl"
    )
    return new_default_group

def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    pg_options_param_name = "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def send_object(obj: Any, dst: int, group: dist.ProcessGroup) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""


        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size

        torch.distributed.send(size_tensor,
                               dst=dst,
                               group=group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=dst,
                               group=group)

        return None

def recv_object(src: int, group: dist.ProcessGroup) -> Any:
    """Receive the input object list from the source rank."""
    """NOTE: `src` is the local rank of the source rank."""


    size_tensor = torch.empty(1, dtype=torch.long)

    # Receive object size
    rank_size = torch.distributed.recv(size_tensor,
                                        src=src,
                                        group=group)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(  # type: ignore[call-overload]
        size_tensor.item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device="cpu")

    rank_object = torch.distributed.recv(object_tensor,
                                            src=src,
                                        group=group)

    assert rank_object == rank_size, (
        "Received object sender rank does not match the size sender rank.")

    obj = pickle.loads(object_tensor.numpy().tobytes())

    return obj

