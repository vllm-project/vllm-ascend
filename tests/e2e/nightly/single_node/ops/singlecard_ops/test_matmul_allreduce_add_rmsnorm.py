import gc
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import torchair

from vllm_ascend.utils import enable_custom_op

config = torchair.CompilerConfig()
config.mode = "reduce-overhead"
npu_backend = torchair.get_npu_backend(compiler_config=config)
torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

global_rank_id = 0


def golden_op_matmul_allreduce_add_rmsnorm(a, b, residual, gamma, epsilon):
    c_ret = torch.nn.functional.linear(a, b)
    dist.all_reduce(c_ret)
    rmsnorm_ret, _, add_ret = torch_npu.npu_add_rms_norm(
        c_ret, residual, gamma, epsilon)
    return rmsnorm_ret, add_ret


def worker(rank, ep_world_size, batch_size, m, k, n):
    global global_rank_id
    global_rank_id = rank
    rank = rank

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="hccl",
                            rank=rank,
                            world_size=ep_world_size)

    ep_ranks_list = list(np.arange(0, ep_world_size))

    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)

    torch_npu.npu.set_device(rank)
    ep_hcomm_info = ep_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(rank)

    torch_npu.npu.synchronize(rank)

    class Module(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

        def forward(self, x1, x2, residual, gamma, ep_hcomm_info, epsilon,
                    is_trans_b, is_allgather_add_out):
            out1, add_out1 = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                x1, x2, residual, gamma, ep_hcomm_info, ep_world_size,
                global_rank_id, epsilon, is_trans_b, is_allgather_add_out)
            return out1, add_out1

    DTYPE = torch.bfloat16
    USE_ONES = False

    torch.manual_seed(42)

    if USE_ONES:
        x1 = torch.ones([m, k], dtype=DTYPE).npu(rank)
        x2 = torch.ones([n, k], dtype=DTYPE).npu(rank)
    else:
        x1 = torch.normal(0, 0.1, [m, k], dtype=DTYPE).npu(rank)
        x2 = torch.normal(0, 0.1, [n, k], dtype=DTYPE).npu(rank)

    if USE_ONES:
        residual = torch.full([m, n], 2048, dtype=DTYPE).npu(rank)
    else:
        residual = torch.full([m, n], 0, dtype=DTYPE).npu(rank)

    gamma = torch.full([n], 1, dtype=DTYPE).npu(rank)

    epsilon = 1e-5
    is_trans_b = True
    is_allgather_add_out = True
    warnup_cnt = 5
    repeat_cnt = 10

    def run_golden_case(loop_cnt):
        for _ in range(loop_cnt):
            golden_out, golden_add_out = golden_op_matmul_allreduce_add_rmsnorm(
                x1, x2, residual, gamma, epsilon)
        torch_npu.npu.synchronize(rank)
        return golden_out, golden_add_out

    run_golden_case(warnup_cnt)

    golden_out, golden_add_out = run_golden_case(repeat_cnt)
    golden_out = golden_out.detach().cpu()
    golden_add_out = golden_add_out.detach().cpu()

    mod = Module().npu()
    opt_model = torch.compile(mod, backend=npu_backend)

    def run_custom_case(loop_cnt):
        for _ in range(loop_cnt):
            out, add_out = opt_model(x1, x2, residual, gamma, ep_hcomm_info,
                                     epsilon, is_trans_b, is_allgather_add_out)
        torch_npu.npu.synchronize(rank)
        return out, add_out

    # warn up
    run_custom_case(warnup_cnt)

    out, add_out = run_custom_case(repeat_cnt)
    out = out.detach().cpu()
    add_out = add_out.detach().cpu()

    dist.destroy_process_group()

    torch.testing.assert_close(golden_out, out, atol=0.1, rtol=0.005)
    torch.testing.assert_close(golden_add_out, add_out, atol=0.1, rtol=0.005)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_matmul_allreduce_add_rmsnorm_kernel():
    ep_world_size = 4
    batch_size = 1
    m = 10000
    k = 1024
    n = 5120
    args = (ep_world_size, batch_size, m, k, n)
    mp.spawn(worker, args=args, nprocs=ep_world_size, join=True)
