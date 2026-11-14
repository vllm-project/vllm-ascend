import random
import socket

import pytest
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
        active_num = m // 8

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16)
        weight1 = self.generate_random_tensor((e, k, n // 8), dtype=torch.int32).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 // 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = bias1.sum(dim=-1).npu()  # shape: [e, n]
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        bias2_npu = bias2.sum(dim=-1).npu()  # shape: [e, n2]

        expert_idx = torch.randint(0, self.world_size * e, (m, topk), dtype=torch.int32)
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64)
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64)
        probs = torch.randn(size=(m, topk), dtype=torch.float32)

        x_active_mask = torch.cat(
            [
                torch.ones(active_num, dtype=torch.bool),
                torch.zeros(m - active_num, dtype=torch.bool),
            ]
        )
        x[active_num:, :] = 0
        expert_idx[active_num:, :] = torch.arange(topk, dtype=torch.int32)

        x = x.npu()
        expert_idx = expert_idx.npu()
        scale1 = scale1.npu()
        scale2 = scale2.npu()
        probs = probs.npu()
        x_active_mask = x_active_mask.npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        bias1_list = []
        bias2_list = []
        for i in range(e):
            weight1_nz_npu.append(torch_npu.npu_format_cast(weight1[i].npu(), 29))
            scale1_npu.append(scale1[i].npu())
            bias1_list.append(bias1_npu[i])

            weight2_nz_npu.append(torch_npu.npu_format_cast(weight2[i].npu(), 29))
            scale2_npu.append(scale2[i].npu())
            bias2_list.append(bias2_npu[i])

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            bias1=bias1_list,
            bias2=bias2_list,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            world_size=self.world_size,
            out=out,
            expert_token_nums=expert_token_nums,
            x_active_mask=x_active_mask,
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
        active_num = m // 2

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16)
        weight1 = self.generate_random_tensor((e, k, n // 8), dtype=torch.int32).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 // 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = bias1.sum(dim=-1).npu()  # shape: [e, n]
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        bias2_npu = bias2.sum(dim=-1).npu()  # shape: [e, n2]

        expert_idx = torch.randint(0, self.world_size * e, (m, topk), dtype=torch.int32)
        scale1 = torch.randint(0, 1, (e, n), dtype=torch.int64)
        scale2 = torch.randint(0, 1, (e, n2), dtype=torch.int64)
        probs = torch.randn(size=(m, topk), dtype=torch.float32)

        x_active_mask = torch.cat(
            [
                torch.ones(active_num, dtype=torch.bool),
                torch.zeros(m - active_num, dtype=torch.bool),
            ]
        )
        x[active_num:, :] = 0
        expert_idx[active_num:, :] = torch.arange(topk, dtype=torch.int32)

        x = x.npu()
        expert_idx = expert_idx.npu()
        scale1 = scale1.npu()
        scale2 = scale2.npu()
        probs = probs.npu()
        x_active_mask = x_active_mask.npu()

        weight1_nz_npu = []
        weight2_nz_npu = []
        scale1_npu = []
        scale2_npu = []
        bias1_list = []
        bias2_list = []

        weight1_nz_npu.append(torch_npu.npu_format_cast(weight1.npu(), 29))
        scale1_npu.append(scale1.npu())
        bias1_list.append(bias1_npu)

        weight2_nz_npu.append(torch_npu.npu_format_cast(weight2.npu(), 29))
        scale2_npu.append(scale2.npu())
        bias2_list.append(bias2_npu)

        out = self.generate_random_tensor((m, k), dtype=torch.bfloat16).npu()
        expert_token_nums = self.generate_random_tensor((1, e), dtype=torch.int32).npu()

        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x,
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx,
            scale1=scale1_npu,
            scale2=scale2_npu,
            bias1=bias1_list,
            bias2=bias2_list,
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            world_size=self.world_size,
            out=out,
            expert_token_nums=expert_token_nums,
            x_active_mask=x_active_mask,
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


def _capacity_worker(rank, world_size, port, active_num, capacity, tensor_list):
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    try:
        with torch.inference_mode():
            _check_logical_token_capacity(op, active_num, capacity, tensor_list)
    finally:
        dist.destroy_process_group()


def _check_logical_token_capacity(op, active_num, capacity, tensor_list):
    num_tokens, hidden_size, ffn_size, num_experts = 64, 1024, 1024, 8
    topk = num_experts
    torch_npu.npu.config.allow_internal_format = True

    # Every expert implements the same normalized projections. This gives a
    # nonzero, bounded reference: silu(x) * x for each constant-valued row.
    values = 1 + (torch.arange(num_tokens, dtype=torch.float32) + op.rank) % 3 / 4
    x = values[:, None].expand(-1, hidden_size).to(torch.bfloat16).contiguous().npu()
    active_mask = (torch.arange(num_tokens) < active_num).npu()
    # Alternate destination ranks; even an odd active_num gives each expert
    # exactly active_num tokens after combining the two source ranks.
    destinations = (torch.arange(num_tokens) + op.rank) % op.world_size
    expert_idx = (destinations[:, None] * num_experts + torch.arange(topk)).to(torch.int32).npu()
    probs = torch.full((num_tokens, topk), 1 / topk, dtype=torch.float32).npu()

    def projection(input_size, output_size):
        # Eight signed int4 values of +1 packed into each int32.
        packed_ones = 0x11111111
        weight = torch.full((num_experts, input_size, output_size // 8), packed_ones, dtype=torch.int32)
        scale = torch.full((num_experts, output_size), 1 / input_size, dtype=torch.float32)
        scale = scale.view(torch.int32).to(torch.int64)
        # W4A8's unsigned low-nibble conversion needs 8 * sum(weight * scale).
        bias = torch.full((num_experts, output_size), 8.0, dtype=torch.float32)
        if tensor_list:
            return (
                [torch_npu.npu_format_cast(item.npu(), 29) for item in weight.unbind()],
                [item.npu() for item in scale.unbind()],
                [item.npu() for item in bias.unbind()],
            )
        return [torch_npu.npu_format_cast(weight.npu(), 29)], [scale.npu()], [bias.npu()]

    weight1, scale1, bias1 = projection(hidden_size, ffn_size)
    weight2, scale2, bias2 = projection(ffn_size // 2, hidden_size)

    def run(inputs, max_output_size):
        out = torch.full_like(inputs, float("nan"))
        counts = torch.full((1, num_experts), -1, dtype=torch.int32, device=inputs.device)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=inputs,
            weight1=weight1,
            weight2=weight2,
            expert_idx=expert_idx,
            scale1=scale1,
            scale2=scale2,
            bias1=bias1,
            bias2=bias2,
            probs=probs,
            group=op.hcomm_info,
            max_output_size=max_output_size,
            world_size=op.world_size,
            out=out,
            expert_token_nums=counts,
            x_active_mask=active_mask,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(counts.cpu(), torch.full((1, num_experts), active_num, dtype=torch.int32))
        return out.cpu()

    reference_capacity = 2 * num_tokens * topk
    reference = run(x, reference_capacity)
    expected = torch.nn.functional.silu(values) * values
    expected[active_num:] = 0
    expected = expected[:, None].expand(-1, hidden_size).to(torch.bfloat16)
    torch.testing.assert_close(reference, expected, rtol=0.02, atol=0.005)

    # Poison shared output buffers with different valid values so missing
    # Combine writes cannot accidentally reuse the reference result.
    run(x * 2, reference_capacity)
    actual = run(x, capacity)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("tensor_list", [False, True], ids=["stacked", "tensor-list"])
@pytest.mark.parametrize(
    "active_num,capacity",
    [(32, 512), (35, 512), (64, 512), (64, 640)],
    ids=["half-full", "expanded-prefix-crosses-capacity", "exact-capacity", "spare-capacity"],
)
def test_dispatch_ffn_combine_logical_token_capacity(active_num, capacity, tensor_list):
    """Changing sufficient logical capacity must not drop any expert output."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    world_size = 2
    mp.spawn(
        _capacity_worker,
        args=(world_size, port, active_num, capacity, tensor_list),
        nprocs=world_size,
        join=True,
    )
