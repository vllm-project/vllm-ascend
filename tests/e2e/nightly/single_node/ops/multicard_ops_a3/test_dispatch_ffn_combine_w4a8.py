import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEVICE_OFFSET = 0


def int32_to_8x_int4_float(tensor_int32):
    """
    Unpack each int32 value in the tensor into 8 signed int4 values and convert them to float32.

    Logic:
    1. Extract the lower 4 bits -> 0th int4
    2. Shift right by 4 bits, extract the lower 4 bits -> 1st int4
    ...
    3. Shift right by 28 bits, extract the lower 4 bits -> 7th int4

    For signed int4 (Two's complement):
    Binary 0000 ~ 0111 (0~7)  -> float 0.0 ~ 7.0
    Binary 1000 ~ 1111 (8~15) -> float -8.0 ~ -1.0
    """

    # Ensure the dtype is int32 (for robustness, even if the input is already int32)
    if tensor_int32.dtype != torch.int32:
        tensor_int32 = tensor_int32.to(torch.int32)

    original_shape = tensor_int32.shape

    # 1. Create shift amounts [0, 4, 8, 12, 16, 20, 24, 28]
    # Reshape to (1, 1, ..., 8) for broadcasting
    shifts = torch.arange(0, 32, 4, device=tensor_int32.device).view(*([1] * len(original_shape)), -1)

    # 2. Expand dimension and shift right
    # unsqueeze(-1) adds a dimension -> [..., 1]
    # After shifting -> [..., 8]
    shifted = tensor_int32.unsqueeze(-1) >> shifts

    # 3. Apply mask to keep only the lower 4 bits (0xF = 1111 binary)
    # The value range here is 0 ~ 15 (unsigned view)
    unpacked_unsigned = shifted & 0xF

    # 4. Convert to signed int4 (-8 ~ 7)
    # If value >= 8, the highest bit is 1, representing a negative number.
    # In two's complement, 4-bit values 8~15 correspond to -8~-1.
    # Algorithm: val = val - 16 (if val >= 8)
    unpacked_signed = unpacked_unsigned.to(torch.int32)  # Ensure calculation precision
    mask = unpacked_signed >= 8
    unpacked_signed[mask] -= 16

    # 5. Convert to float32
    result_float = unpacked_signed.to(torch.float32)
    result_flat = result_float.flatten(start_dim=-2)
    return result_flat


class TestDispatchFFNCombine:
    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port

    def get_hcomm(self, comm_group):
        hcomm_info = None
        if torch.__version__ > "2.0.1":
            hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
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
        torch_npu.npu.set_device(DEVICE_OFFSET + self.rank)
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
            group_ep, group_tp = self.setup_ep_tp(self.rank, tp_size, ep_size, "hccl", None, None)
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

    def run_tensor_list(self) -> bool:
        torch_npu.npu.set_device(DEVICE_OFFSET + self.rank)
        m = 64
        k = 1024
        n = 1024
        topk = 8
        e = 8
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = self.generate_random_tensor((e, k, n // 8), dtype=torch.int32).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 // 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = bias1.sum(dim=-1).npu()
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        bias2_npu = bias2.sum(dim=-1).npu()

        print("====generate bias====")
        expert_idx = torch.randint(0, self.world_size * e, (m, topk), dtype=torch.int32).npu()
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()
        probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        for i in range(e):
            weight1_nz_npu.append(torch_npu.npu_format_cast(weight1[i].npu(), 29))
            scale1_npu.append(scale1[i].npu())
            weight2_nz_npu.append(torch_npu.npu_format_cast(weight2[i].npu(), 29))
            scale2_npu.append(scale2[i].npu())

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            bias1=bias1_npu,
            bias2=bias2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        return True

    def run_normal(self) -> bool:
        torch_npu.npu.set_device(DEVICE_OFFSET + self.rank)
        m = 64
        k = 1024
        n = 1024
        topk = 8
        e = 8
        k2 = n // 2
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        weight1 = self.generate_random_tensor((e, k, n / 8), dtype=torch.int32).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 / 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        expert_idx = torch.randint(0, self.world_size * e, (m, topk), dtype=torch.int32).npu()
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64).npu()
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64).npu()
        probs = torch.randn(size=(m, topk), dtype=torch.float32).npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        weight1_nz_npu.append(torch_npu.npu_format_cast(weight1.npu(), 29))
        scale1_npu.append(scale1.npu())
        weight2_nz_npu.append(torch_npu.npu_format_cast(weight2.npu(), 29))
        scale2_npu.append(scale2.npu())

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()

        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        return True

    def generate_random_tensor(self, size, dtype):
        if dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return torch.randn(size=size, dtype=dtype)
        elif dtype is torch.int8:
            return torch.randint(-16, 16, size=size, dtype=dtype)
        elif dtype is torch.int32:
            return torch.randint(-127, 127, size=size, dtype=dtype)
        else:
            raise ValueError(f"Invalid dtype: {dtype}")


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    out1 = op.run_tensor_list()
    q.put(out1)
    out2 = op.run_normal()
    q.put(out2)


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
