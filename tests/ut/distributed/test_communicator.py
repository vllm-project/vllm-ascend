import functools
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.distributed.communicator import NPUCommunicator

world_size = 2


def npu_distributed_test():

    def decorator(test_func):

        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            if not os.environ.get("LOCAL_RANK"):
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = "29500"
                os.environ["WORLD_SIZE"] = str(world_size)
                mp.spawn(_distributed_worker,
                         args=(world_size, test_func.__name__, *args),
                         nprocs=world_size,
                         join=True)

        return wrapper

    return decorator


def get_innermost_func(decorated_func):
    while hasattr(decorated_func, '__wrapped__'):
        decorated_func = decorated_func.__wrapped__
    return decorated_func


def _distributed_worker(rank, world_size, test_func_name, *args, **kwargs):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="hccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=rank,
                                timeout=timedelta(seconds=30))

    torch.npu.set_device(rank)
    test_func = getattr(TestNPUCommunicator, test_func_name)
    test_func = get_innermost_func(test_func)
    test_func(*args, **kwargs)

    dist.destroy_process_group()
    torch.npu.empty_cache()


class TestNPUCommunicator:
    @npu_distributed_test()
    def test_all_to_all_with_sizes(self) -> torch.Tensor:
        rank = dist.get_rank()
        torch.npu.set_device(rank)
        device = torch.device("npu", index=rank)

        if rank == 0:
            scatter_sizes = [2, 1]
            gather_sizes = [2, 1]
            input_ = torch.tensor([10, 20, 30], device=device)
        else:
            scatter_sizes = [1, 2]
            gather_sizes = [1, 2]
            input_ = torch.tensor([40, 50, 60], device=device)

        comm = NPUCommunicator(cpu_group=dist.group.WORLD,
                               device=device,
                               device_group=dist.group.WORLD)

        output = comm.all_to_all(input_,
                                 scatter_sizes=scatter_sizes,
                                gather_sizes=gather_sizes)

        dist.barrier()

        if rank == 0:
            assert output.tolist() == [10, 20, 40]
        else:
            assert output.tolist() == [30, 50, 60]

    @npu_distributed_test()
    def test_all_to_all_without_sizes(self) -> torch.Tensor:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("npu", rank)

        # rank 0: [[1, 2], [101, 102]]
        # rank 1: [[11, 12], [111, 112]]
        input_data = [[rank * 10 + r * 100 + 1, rank * 10 + r * 100 + 2]
                       for r in range(world_size)]
        input_ = torch.tensor(input_data, device=device)

        comm = NPUCommunicator(cpu_group=dist.group.WORLD,
                               device_group=dist.group.WORLD,
                               device=device)
        output = comm.all_to_all(input_, scatter_dim=0, gather_dim=0)

        dist.barrier()

        if rank == 0:
            assert output.tolist() == [[1, 2], [11, 12]]
        else:
            assert output.tolist() == [[101, 102], [111, 112]]