import os
import random
import subprocess
import sys
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEVICE_OFFSET = 0

BASE_KWARGS = {
    "batch_size": 64,
    "token_hidden_size": 1024,
    "moe_intermediate_size": 512,  # post-SwiGLU dim; GMM1 outputs 2× this, GMM2 takes this as input
    "top_k": 8,
    "moe_expert_num": 16,  # global expert count; per-rank = this // ep_world_size
    "ep_world_size": 2,
    "active_ratio_tensor_list": 1,  # m // 1 = all tokens active
    "active_ratio_normal": 1,       # m // 1 = all tokens active
    "output_dir": "./output",
    "profile": False,
}

# 1C2V mix kernel: infer upper bound for profiling GM buffer (see dispatch_ffn_combine_w4_a8_proto.cpp)
MAX_INFER_GETBLOCKNUM_UB = 128
MIX_AIC_1_2_SLOTS_PER_GROUP = 3
PROF_SIZE_PER_CORE = 2048
PROFILING_NUMEL = MAX_INFER_GETBLOCKNUM_UB * MIX_AIC_1_2_SLOTS_PER_GROUP * PROF_SIZE_PER_CORE

VLLM_ASCEND_ROOT = Path(__file__).resolve().parents[6]
TRACE_SCRIPT_DIR = VLLM_ASCEND_ROOT / "csrc" / "scripts" / "trace"
OP_KERNEL_DIR = VLLM_ASCEND_ROOT / "csrc" / "mc2" / "dispatch_ffn_combine_w4_a8" / "op_kernel"
BASE_H_PATH = OP_KERNEL_DIR / "dispatch_ffn_combine_w4_a8_base.h"
TRACE_PREPROCESSOR = TRACE_SCRIPT_DIR / "trace_preprocessor.py"
TRACE_COLLECTOR = TRACE_SCRIPT_DIR / "trace_collector.py"


def _ensure_point_map(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_map = output_dir / "point_map.json"
    if point_map.is_file():
        return point_map
    if not TRACE_PREPROCESSOR.is_file():
        raise FileNotFoundError(f"trace_preprocessor not found: {TRACE_PREPROCESSOR}")
    subprocess.run(
        [sys.executable, str(TRACE_PREPROCESSOR), str(OP_KERNEL_DIR), str(output_dir)],
        check=True,
    )
    if not point_map.is_file():
        raise FileNotFoundError(f"failed to generate point_map.json under {output_dir}")
    return point_map


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
    def __init__(self, rank, world_size, port, kwargs):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port
        self.kwargs = kwargs

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
        m = self.kwargs["batch_size"]
        k = self.kwargs["token_hidden_size"]
        n = self.kwargs["moe_intermediate_size"]
        topk = self.kwargs["top_k"]
        e_global = self.kwargs["moe_expert_num"]  # global expert count
        e = e_global // self.world_size            # per-rank expert count
        k2 = n  # GMM2 input: post-SwiGLU intermediate_size
        n2 = k  # GMM2 output: token_hidden_size
        active_num = m // self.kwargs["active_ratio_tensor_list"]

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16)
        weight1 = self.generate_random_tensor((e, k, n * 2 // 8), dtype=torch.int32).npu()  # GMM1: K × 2*intermediate
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 // 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = bias1.sum(dim=-1).npu()
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        bias2_npu = bias2.sum(dim=-1).npu()

        expert_idx = torch.arange(
            self.rank * m * topk,
            self.rank * m * topk + m * topk,
            dtype=torch.int32).view(m, topk) % e_global
        scale1 = torch.randint(0, 1, (e, n * 2), dtype=torch.int64)  # GMM1 output: 2*intermediate channels
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
            out=out,
            expert_token_nums=expert_token_nums,
            x_active_mask=x_active_mask,
        )
        return True

    def run_normal(self, profiling_dir: Path) -> bool:
        torch_npu.npu.set_device(DEVICE_OFFSET + self.rank)
        m = self.kwargs["batch_size"]
        k = self.kwargs["token_hidden_size"]
        n = self.kwargs["moe_intermediate_size"]
        topk = self.kwargs["top_k"]
        e_global = self.kwargs["moe_expert_num"]  # global expert count
        e = e_global // self.world_size            # per-rank expert count
        k2 = n  # GMM2 input: post-SwiGLU intermediate_size
        n2 = k  # GMM2 output: token_hidden_size
        active_num = m // self.kwargs["active_ratio_normal"]

        torch_npu.npu.config.allow_internal_format = True
        x = self.generate_random_tensor((m, k), dtype=torch.bfloat16)
        weight1 = self.generate_random_tensor((e, k, n * 2 // 8), dtype=torch.int32).npu()  # GMM1: K × 2*intermediate
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2 // 8), dtype=torch.int32).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

        bias1 = int32_to_8x_int4_float(weight1.cpu())
        bias1_npu = bias1.sum(dim=-1).npu()
        bias2 = int32_to_8x_int4_float(weight2.cpu())
        bias2_npu = bias2.sum(dim=-1).npu()

        expert_idx = torch.arange(
            self.rank * m * topk,
            self.rank * m * topk + m * topk,
            dtype=torch.int32).view(m, topk) % e_global
        scale1 = torch.randint(0, 1, (e, n * 2), dtype=torch.int64)  # GMM1 output: 2*intermediate channels
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

        profiling_data = torch.zeros(PROFILING_NUMEL, dtype=torch.int64, device=f"npu:{DEVICE_OFFSET + self.rank}")
        profiling_data_last = torch.zeros(PROFILING_NUMEL, dtype=torch.int64, device=f"npu:{DEVICE_OFFSET + self.rank}")

        for i in range(100):
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
                out=out,
                expert_token_nums=expert_token_nums,
                x_active_mask=x_active_mask,
                profiling_data=(profiling_data_last if i == 99 else profiling_data),
            )
            torch_npu.npu.synchronize()

        sys.path.insert(0, str(TRACE_SCRIPT_DIR))
        import trace_utils

        trace_utils.save_profiling_data(
            profiling_data_last.cpu(),
            self.rank,
            str(profiling_dir),
            base_h_path=str(BASE_H_PATH.resolve()),
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


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue, kwargs: dict):
    op = TestDispatchFFNCombine(rank, world_size, port, kwargs)
    op.generate_hcom()
    out1 = op.run_tensor_list()
    q.put(out1)
    out2 = op.run_normal()
    q.put(out2)


def worker_profiling(rank: int, world_size: int, port: int, profiling_dir: str, q: mp.SimpleQueue, kwargs: dict):
    op = TestDispatchFFNCombine(rank, world_size, port, kwargs)
    op.generate_hcom()
    ok = op.run_normal(Path(profiling_dir))
    q.put(ok)


@torch.inference_mode()
def test_dispatch_ffn_combine_kernel(kwargs: dict = None):
    if kwargs is None:
        kwargs = BASE_KWARGS
    world_size = kwargs["ep_world_size"]
    mp.set_start_method("fork", force=True)

    q = mp.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, port, q, kwargs))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]

    for p in p_list:
        p.join()

    assert all(results)


@torch.inference_mode()
def test_dispatch_ffn_combine_w4a8_profiling(kwargs: dict = None):
    """W4A8 operator run with profiling_data + rank*.pt dump (+ optional Chrome trace)."""
    if kwargs is None:
        kwargs = BASE_KWARGS
    world_size = kwargs["ep_world_size"]
    mp.set_start_method("fork", force=True)

    profiling_dir = Path(
        os.environ.get(
            "DFFC_PROFILING_DIR",
            "/tmp/dispatch_ffn_combine_w4a8_profiling",
        )
    ).expanduser().resolve()
    profiling_dir.mkdir(parents=True, exist_ok=True)

    point_map_path = (
        Path(os.environ["DFFC_POINT_MAP"]).expanduser().resolve()
        if os.environ.get("DFFC_POINT_MAP")
        else _ensure_point_map(profiling_dir)
    )
    chrome_trace_path = (
        Path(os.environ["DFFC_CHROME_TRACE"]).expanduser().resolve()
        if os.environ.get("DFFC_CHROME_TRACE")
        else profiling_dir / "chrome_trace.json"
    )

    q = mp.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = mp.Process(
            target=worker_profiling,
            args=(rank, world_size, port, str(profiling_dir), q, kwargs),
        )
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]
    for p in p_list:
        p.join()

    assert all(results)

    rank_pts = sorted(profiling_dir.glob("rank*.pt"))
    assert rank_pts, f"no rank*.pt under {profiling_dir}"

    if TRACE_COLLECTOR.is_file():
        subprocess.run(
            [
                sys.executable,
                str(TRACE_COLLECTOR),
                str(profiling_dir),
                str(point_map_path),
                "-o",
                str(chrome_trace_path),
            ],
            check=True,
        )
        assert chrome_trace_path.is_file(), f"chrome trace not generated: {chrome_trace_path}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test dispatch_ffn_combine w4a8 kernel with configurable parameters"
    )
    parser.add_argument("--batch_size", type=int, default=BASE_KWARGS["batch_size"],
                        help="Batch/token count (default: 64)")
    parser.add_argument("--token_hidden_size", type=int, default=BASE_KWARGS["token_hidden_size"],
                        help="Input hidden size (default: 1024)")
    parser.add_argument("--moe_intermediate_size", type=int, default=BASE_KWARGS["moe_intermediate_size"],
                        help="Intermediate size — post-SwiGLU dim (default: 512)")
    parser.add_argument("--top_k", type=int, default=BASE_KWARGS["top_k"],
                        help="Top-k experts (default: 8)")
    parser.add_argument("--moe_expert_num", type=int, default=BASE_KWARGS["moe_expert_num"],
                        help="Global expert count across all ranks (default: 16)")
    parser.add_argument("--ep_world_size", type=int, default=BASE_KWARGS["ep_world_size"],
                        help="Number of processes (default: 2)")
    parser.add_argument("--active_ratio_tensor_list", type=int, default=BASE_KWARGS["active_ratio_tensor_list"],
                        help="Denominator for active_num in run_tensor_list, m // N (default: 1)")
    parser.add_argument("--active_ratio_normal", type=int, default=BASE_KWARGS["active_ratio_normal"],
                        help="Denominator for active_num in run_normal, m // N (default: 1)")
    # torch_npu.profile功能
    parser.add_argument("--output_dir", type=str, default=BASE_KWARGS["output_dir"],
                        help="Output directory for result storage (default: ./output)")

    # 打点功能
    parser.add_argument("--profiling", action="store_true", help="run profiling test only")
    parser.add_argument("--profiling_dir", type=str, default="/tmp/dispatch_ffn_combine_w4a8_profiling")
    parser.add_argument("--point_map", type=str, default=None)
    parser.add_argument("--chrome_trace", type=str, default=None)

    args = parser.parse_args()
    BASE_KWARGS["batch_size"] = args.batch_size
    BASE_KWARGS["token_hidden_size"] = args.token_hidden_size
    BASE_KWARGS["moe_intermediate_size"] = args.moe_intermediate_size
    BASE_KWARGS["top_k"] = args.top_k
    BASE_KWARGS["moe_expert_num"] = args.moe_expert_num
    BASE_KWARGS["ep_world_size"] = args.ep_world_size

    BASE_KWARGS["active_ratio_tensor_list"] = args.active_ratio_tensor_list
    BASE_KWARGS["active_ratio_normal"] = args.active_ratio_normal
    # torch_npu.profile功能
    BASE_KWARGS["output_dir"] = args.output_dir
    BASE_KWARGS["profile"] = True

    if args.profiling_dir:
        os.environ["DFFC_PROFILING_DIR"] = str(Path(args.profiling_dir).expanduser().resolve())
    if args.point_map:
        os.environ["DFFC_POINT_MAP"] = str(Path(args.point_map).expanduser().resolve())
    if args.chrome_trace:
        os.environ["DFFC_CHROME_TRACE"] = str(Path(args.chrome_trace).expanduser().resolve())

    if args.profiling:
        test_dispatch_ffn_combine_w4a8_profiling(BASE_KWARGS)
    else:
        test_dispatch_ffn_combine_kernel(BASE_KWARGS)

    print("test_dispatch_ffn_combine_kernel PASSED")
