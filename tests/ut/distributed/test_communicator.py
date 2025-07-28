import multiprocessing
import os
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from vllm_ascend.distributed.communicator import NPUCommunicator


def update_environment_variables(envs: dict[str, str]):
    print(f"Initializing distributed environment with {envs=}")
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            print(
                "Overwriting environment variable %s "
                "from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def distributed_run(fn, world_size, *args, **kwargs):
    number_of_processes = world_size
    processes: list[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12345"
        # _init_distributed_environment.decorator == fn
        p = multiprocessing.Process(target=fn,
                                    args=(env, *args),
                                    kwargs=kwargs)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def _init_distributed_environment(fn):

    def decorator(env, *args, **kwargs):
        update_environment_variables(env)
        with PatchTorchDistributed(fn):
            fn(*args, **kwargs)

    return decorator


def test_all_to_all_with_sizes():
    distributed_run(all_to_all_with_sizes, world_size=2)


def test_all_to_all_without_sizes():
    distributed_run(all_to_all_without_sizes, world_size=2)


class PatchTorchDistributed:

    def __init__(self, func):
        rank = int(os.getenv("RANK", 0))
        self.mock_rank_patcher = patch("torch.distributed.get_rank",
                                       return_value=rank)
        self.mock_world_size_patcher = patch(
            "torch.distributed.get_world_size",
            return_value=int(os.getenv("WORLD_SIZE", 0)),
        )

        def patched_all_to_all(output_tensor_list,
                               input_tensor_list,
                               group=None,
                               async_op=False):
            if func.__name__ == "all_to_all_with_sizes":
                output_tensor_list[:] = (
                    [torch.tensor([10, 20]),
                     torch.tensor([50, 60])] if rank == 0 else
                    [torch.tensor([30, 40]),
                     torch.tensor([70, 80])])
            elif func.__name__ == "all_to_all_without_sizes":
                output_tensor_list[:] = (
                    [torch.tensor([[10, 20]]),
                     torch.tensor([[50, 60]])] if rank == 0 else
                    [torch.tensor([[30, 40]]),
                     torch.tensor([[70, 80]])])

            if async_op:
                mock_work = MagicMock()
                mock_work.wait.return_value = None
                return mock_work
            return

        # rank 0: [10, 20, 30, 40] -> [10, 20, 50, 60]
        # rank 1: [50, 60, 70, 80] -> [30, 40, 70, 80]
        self.mock_all_to_all_patcher = patch(
            "torch.distributed.all_to_all",
            new_callable=lambda: patched_all_to_all)

        # Patch get group ranks
        self.patch_get_pgr = patch("torch.distributed.get_process_group_ranks",
                                   return_value={
                                       0: 0,
                                       1: 1
                                   })
        self.patch_get_gr = patch("torch.distributed.get_group_rank",
                                  return_value={
                                      0: 0,
                                      1: 1
                                  })

        # Patch torch npu
        self.mock_npu_device_patcher = patch("torch.npu.current_device",
                                             return_value=MagicMock())
        self.mock_npu_set_device_patcher = patch("torch.npu.set_device",
                                                 return_value=MagicMock())
        self.mock_torch_device_patcher = patch(
            "torch.device", return_value=torch.device("cpu"))

    def __enter__(self):
        self.mock_rank_patcher.start()
        self.mock_world_size_patcher.start()
        self.mock_all_to_all_patcher.start()

        self.patch_get_pgr.start()
        self.patch_get_gr.start()

        self.mock_npu_device_patcher.start()
        self.mock_npu_set_device_patcher.start()
        self.mock_torch_device_patcher.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mock_rank_patcher.stop()
        self.mock_world_size_patcher.stop()
        self.mock_all_to_all_patcher.stop()

        self.patch_get_pgr.stop()
        self.patch_get_gr.stop()

        self.mock_npu_device_patcher.stop()
        self.mock_npu_set_device_patcher.stop()
        self.mock_torch_device_patcher.stop()


@_init_distributed_environment
def all_to_all_with_sizes():
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    device = torch.device("npu", index=rank)

    if rank == 0:
        scatter_sizes = [2, 2]
        gather_sizes = [2, 2]
        input_ = torch.tensor([10, 20, 30, 40], device=device)
    else:
        scatter_sizes = [2, 2]
        gather_sizes = [2, 2]
        input_ = torch.tensor([50, 60, 70, 80], device=device)

    comm = NPUCommunicator(cpu_group=dist.group.WORLD,
                           device=device,
                           device_group=dist.group.WORLD)

    output = comm.all_to_all(input_,
                             scatter_sizes=scatter_sizes,
                             gather_sizes=gather_sizes)

    if rank == 0:
        assert output.tolist() == [10, 20, 50, 60]
    else:
        assert output.tolist() == [30, 40, 70, 80]

    print("all_to_all_with_sizes", "Rank %d done." % rank, "output:", output)


@_init_distributed_environment
def all_to_all_without_sizes():
    rank = dist.get_rank()
    device = torch.device("npu", rank)

    if rank == 0:
        input_ = torch.tensor([[10, 20], [30, 40]], device=device)
    else:
        input_ = torch.tensor([[50, 60], [70, 80]], device=device)

    comm = NPUCommunicator(cpu_group=dist.group.WORLD,
                           device_group=dist.group.WORLD,
                           device=device)
    output = comm.all_to_all(input_, scatter_dim=0, gather_dim=0)

    if rank == 0:
        assert output.tolist() == [[10, 20], [50, 60]]
    else:
        assert output.tolist() == [[30, 40], [70, 80]]

    print("all_to_all_without_sizes", "Rank %d done." % rank, "output:",
          output)
