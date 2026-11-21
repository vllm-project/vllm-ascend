import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

DEVICE_OFFSET = 0


def _is_a2() -> bool:
    """A2 (ascend910b) has no HCCL symmetric window and needs the KFC context."""
    try:
        from vllm_ascend.device.device_config import get_ascend_device_type
        from vllm_ascend.device.hardware import AscendDeviceType

        return get_ascend_device_type() == AscendDeviceType.A2
    except Exception:  # noqa: BLE001 - A3 runs must not fail on device probing
        return False


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
            device_id=torch.device("npu", DEVICE_OFFSET + self.rank),
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
                group_mc2 = dist.new_group(
                    ranks=list(range(self.world_size)),
                    backend="hccl",
                    group_desc="mc2:device",
                    device_id=torch.device("npu", DEVICE_OFFSET + self.rank),
                )
                warm = torch.ones(1, dtype=torch.float32).npu()
                dist.all_reduce(warm, group=group_mc2)
                torch.npu.synchronize()
                hcomm_info_dist["default_pg_info"] = self.get_hcomm(group_mc2)

                # A2 KFC context: symmetric CCL buffers (epHcclBuffer_). A3 keeps
                # the HCCL symmetric window, so context stays None there.
                self.kfc_context = None
                self.ccl_buffer_size = 0
                if _is_a2():
                    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe

                    self.sym_buffer = get_symm_buffer_for_mega_moe(
                        group_mc2,
                        num_experts=8,
                        num_max_tokens_per_rank=128,
                        num_topk=8,
                        hidden=1024,
                        intermediate_hidden=512,
                    )
                    self.kfc_context = self.sym_buffer.context
                    self.ccl_buffer_size = self.sym_buffer.ccl_buffer_size
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
        weight1 = self.generate_random_tensor((e, k, n), dtype=torch.bfloat16).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2), dtype=torch.bfloat16).npu()
        weight2 = torch_npu.npu_format_cast(weight2, 29)

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
            bias1=[],
            bias2=[],
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            world_size=self.world_size,
            context=self.kfc_context,
            ccl_buffer_size=self.ccl_buffer_size,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch.npu.synchronize()
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
        weight1 = self.generate_random_tensor((e, k, n), dtype=torch.bfloat16).npu()
        weight1 = torch_npu.npu_format_cast(weight1, 29)
        weight2 = self.generate_random_tensor((e, k2, n2), dtype=torch.bfloat16).npu()
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
            bias1=[],
            bias2=[],
            probs=probs,
            group=self.hcomm_info,
            max_output_size=512,
            world_size=self.world_size,
            context=self.kfc_context,
            ccl_buffer_size=self.ccl_buffer_size,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch.npu.synchronize()
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
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    try:
        out1 = op.run_tensor_list()
    except Exception as e:  # noqa: BLE001
        print(f"[rank {rank}] tensor_list FAILED: {type(e).__name__}: {e}", flush=True)
        out1 = False
    q.put(("tensor_list", out1))
    try:
        out2 = op.run_normal()
    except Exception as e:  # noqa: BLE001
        print(f"[rank {rank}] normal FAILED: {type(e).__name__}: {e}", flush=True)
        out2 = False
    q.put(("normal", out2))


@torch.inference_mode()
def test_dispatch_ffn_combine_kernel():
    world_size = 2
    mp.set_start_method("spawn", force=True)

    q = mp.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, port, q))
        p.start()
        p_list.append(p)

    # Each worker contributes two results (tensor-list and normal path);
    # retrieve them all so a failure on either path is asserted.
    results = [q.get() for _ in range(world_size * 2)]

    for p in p_list:
        p.join()

    by_path: dict[str, list[bool]] = {}
    for path, ok in results:
        by_path.setdefault(path, []).append(ok)
    for path in ("tensor_list", "normal"):
        assert by_path.get(path), f"missing {path} results: {by_path}"
        assert all(by_path[path]), f"{path} path failed on some rank: {by_path[path]}"
