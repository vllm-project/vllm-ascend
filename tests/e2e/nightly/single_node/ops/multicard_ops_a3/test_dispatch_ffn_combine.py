import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


class TestDisptachFFNCombine:

    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port

    def get_hcomm(self, comm_group):
        hcomm_info = None
        if torch.__version__ > "2.0.1":
            hcomm_info = comm_group._get_backend(
                torch.device("npu")).get_hccl_comm_name(self.rank)
        else:
            hcomm_info = comm_group.get_hccl_comm_name(self.rank)
        return hcomm_info

    def setup_ep_tp(
        self,
        rank,
        tp_size,
        ep_size,
        backend_type,
        ep_ranks_list=None,
        tp_ranks_list=None,
    ):
        for i in range(tp_size):
            if ep_ranks_list:
                ep_ranks = ep_ranks_list[i]
            else:
                ep_ranks = [x + ep_size * i for x in range(ep_size)]
            ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group_tmp = ep_group
        for i in range(ep_size):
            if tp_ranks_list:
                tp_ranks = tp_ranks_list[i]
            else:
                tp_ranks = [x * ep_size + i for x in range(tp_size)]
            tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group_tmp = tp_group
        return ep_group_tmp, tp_group_tmp

    def generate_hcom(self):
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )

        ep_size = 0
        tp_size = self.world_size
        hcomm_info_dist = {
            "default_pg_info": None,
            "ep_hcomm_info": None,
            "group_ep": None,
            "tp_hcomm_info": None,
            "group_tp": None,
        }
        if ep_size and tp_size:
            group_ep, group_tp = self.setup_ep_tp(self.rank, tp_size, ep_size,
                                                  "hccl", None, None)
            hcomm_info_dist["ep_hcomm_info"] = self.get_hcomm(group_ep)
            hcomm_info_dist["tp_hcomm_info"] = self.get_hcomm(group_tp)
            hcomm_info_dist["group_ep"] = group_ep
            hcomm_info_dist["group_tp"] = group_tp
        else:
            if dist.is_available():
                default_pg = _get_default_group()
            hcomm_info_dist["default_pg_info"] = self.get_hcomm(default_pg)
        hcomm_info = hcomm_info_dist["default_pg_info"]
        self.hcomm_info = hcomm_info

    def run_npu_out(self) -> bool:
        torch_npu.npu.set_device(self.rank)
        m = 2  # token-num  32
        k = 4  # hidden_size 7168
        n = 4  # mid-hidden-size  4096
        topk = 2
        e = 2  # expert-num-per-rank  16
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = self.generate_random_tensor((e, k, n),
                                              dtype=torch.int8).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2),
                                              dtype=torch.int8).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        expert_idx = torch.randint(0,
                                   self.world_size * e, (m, topk),
                                   dtype=torch.int32).npu()
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()
        probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()
        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()

        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1,
            weight2=weight2,
            expert_idx=expert_idx,
            scale1=scale1,
            scale2=scale2,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            out=out,
        )
        return True

    def generate_random_tensor(self, size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.randn(size=size, dtype=dtype)
        elif dtype is torch.int8:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        elif dtype is torch.int32:
            return torch.randint(-1024, 1024, size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    op = TestDisptachFFNCombine(rank, world_size, port)
    op.generate_hcom()
    out = op.run_npu_out()
    q.put(out)


@torch.inference_mode()
def test_dispatch_ffn_combine_kernel():
    world_size = 2
    mp.set_start_method("fork", force=True)

    q = mp.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, port, q))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]

    for p in p_list:
        p.join()

    assert all(results)
