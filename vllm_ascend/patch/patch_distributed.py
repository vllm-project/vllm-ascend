import torch
from torch.distributed import ProcessGroup, TCPStore
from torch.distributed.distributed_c10d import (Backend, PrefixStore,
                                                _get_default_timeout,
                                                _unregister_process_group,
                                                is_nccl_available)
from torch.distributed.rendezvous import rendezvous
import vllm
import vllm.distributed
from vllm.config import ParallelConfig


def ascend_destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None
    from vllm.platforms import current_platform
    current_platform.destroy_platform_model_parallel()


def ascend_stateless_init_torch_distributed_process_group(
        host: str, port: int, rank: int, world_size: int,
        backend: str) -> ProcessGroup:
    """
    A replacement for `torch.distributed.init_process_group` that does not
    pollute the global state. The created ProcessGroup object can be used for
    some operations such as `allreduce`, because it does not depend on the
    global rank. However, some operations such as `broadcast` cannot be used
    because it depends on the global rank.

    # TODO: ask for help from PyTorch team if we need the `broadcast` operation.

    This function is useful when we are not sure about the total number of
    processes in the process group. For example, we may have process
    1, 2, ..., 8 who want to communicate, and process 9 might be the same
    process as process 1, or it might be a different process; process 10
    might be the same process as process 5, or it might be a different process.
    In this case, how can we reliably form a communication channel within
    process 9 and 10, without affecting the communication channel within
    process 1, 2, ..., 8?

    One possible solution is to figure out if process 9 and 10 are the same
    as process 1 and 5 beforehand, and then form a communication channel
    based on the information, adjusting the ranks and world_size etc. However,
    figuring out the information is not always easy, and it will interfere
    with the main communication channel.

    Our solution is to always form a communication channel with process 1, 2,
    ..., 8, and then use this function to form another communication channel
    with process 9 and 10. This way, regardless of whether process 9 and 10
    are the same as process 1 and 5, the main communication channel is
    always formed with process 1, 2, ..., 8, and the additional communication
    channel is formed with process 9 and 10.
    """
    init_method = f"tcp://{host}:{port}"
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout))
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
    )
    from vllm.platforms import current_platform
    if backend == "gloo":
        from torch.distributed.distributed_c10d import ProcessGroupGloo
        backend_class = ProcessGroupGloo(prefix_store,
                                         group_rank,
                                         group_size,
                                         timeout=timeout)
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
    elif backend == "nccl":
        assert is_nccl_available()
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        backend_options = ProcessGroupNCCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupNCCL(prefix_store, group_rank, group_size,
                                         backend_options)
        backend_type = ProcessGroup.BackendType.NCCL
        device = torch.device("cuda")
    elif current_platform.platform_has_backend_register():
        current_platform.platform_register_backend()
        return pg
    else:
        raise RuntimeError(f"Unsupported torch distributed backend: {backend}")

    pg._set_default_backend(backend_type)
    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)

    return pg

def parallel_config_get_dp_port(self) -> int:
    """
    We might need to initialize process groups in multiple
    processes that is related to data parallelism,
    e.g. both in the worker and in the engine, which
    can live in different processes. To avoid port conflicts, we
    increment the port number each time we need to initialize a
    new process group related to data parallelism.
    """
    answer = self.data_parallel_master_port
    self.data_parallel_master_port += 1
    import os
    # NOTE: Get port from envs directly when using torchrun
    port = os.environ.get("MASTER_PORT", answer)
    return port

vllm.distributed.parallel_state.destroy_model_parallel = ascend_destroy_model_parallel
vllm.distributed.stateless_init_torch_distributed_process_group = ascend_stateless_init_torch_distributed_process_group
ParallelConfig.get_next_dp_init_port = parallel_config_get_dp_port