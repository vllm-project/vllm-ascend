import random
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

enable_custom_op()

try:
    if not torch.compiler.is_compiling():
        bootstrap_custom_op_env()
    # isort: off
    # register custom ops into torch_library here
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

    # register the meta implementation for custom kernel if necessary
    # import vllm_ascend.meta_registration  # type: ignore  # noqa: F401
except ImportError:
    pass

from vllm_ascend.ops.mega_moe import get_symm_buffer_for_mega_moe, mega_moe
def _ceil(a, b):
    return (a + b - 1) // b


def _get_float8_e8m0_dtype():
    for module, name in (
        (torch, "float8_e8m0fnu"),
        (torch, "float8_e8m0"),
        (torch_npu, "float8_e8m0fnu"),
        (torch_npu, "float8_e8m0"),
    ):
        if hasattr(module, name):
            return getattr(module, name)
    raise RuntimeError("float8_e8m0 dtype is not available")


class MegaMoeRunner:

    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        self.master_ip = "127.0.0.1"
        self.port = port
        self.ep_group = None

    def generate_hcom(self):
        torch_npu.npu.set_device(self.rank)
        dist.init_process_group(
            backend="hccl",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        self.ep_group = dist.new_group(backend="hccl", ranks=list(range(self.world_size)))

        hcoom_info = self.ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)

        assert hcoom_info, "Failed to get a valid HCCL comm name for mega_moe EP group."

    @staticmethod
    def check_output(y, expert_token_nums, x, local_expert_num) -> bool:
        torch_npu.npu.synchronize()
        return (
            y.shape == x.shape
            and y.dtype == x.dtype
            and expert_token_nums.shape == (local_expert_num,)
            and expert_token_nums.dtype == torch.int32
        )

    def _make_quant_inputs(self, weight_dtype):
        seed = 1234
        torch.manual_seed(seed)
        torch_npu.npu.manual_seed(seed)
        bs, hidden, num_topk, local_expert_num, intermediate = 256, 4096, 6, 4, 1024
        num_experts = local_expert_num * self.world_size
        intermediate_per_gate = intermediate // 2

        x = torch.randn(bs, hidden, dtype=torch.bfloat16).npu()
        topk_ids = torch.stack([
            torch.randperm(num_experts)[:num_topk] for _ in range(bs)
        ]).to(torch.int32).npu()
        topk_weights = torch.randn(bs, num_topk, dtype=torch.bfloat16).npu()

        weight1 = torch.randn(
            local_expert_num, intermediate, hidden, dtype=torch.float32
        ).to(weight_dtype).npu()
        weight2 = torch.randn(
            local_expert_num, hidden, intermediate_per_gate, dtype=torch.float32
        ).to(weight_dtype).npu()

        fp8_e8m0 = _get_float8_e8m0_dtype()
        w1_scale_shape = (local_expert_num, intermediate, _ceil(hidden, 64), 2)
        w2_scale_shape = (
            local_expert_num,
            hidden,
            _ceil(intermediate_per_gate, 64),
            2,
        )
        w1_scales = torch.randint(
            125, 130, w1_scale_shape, dtype=torch.uint8
        ).view(fp8_e8m0).npu()
        w2_scales = torch.randint(
            125, 130, w2_scale_shape, dtype=torch.uint8
        ).view(fp8_e8m0).npu()

        return (
            x,
            topk_ids,
            topk_weights,
            [weight1],
            [weight2],
            [w1_scales],
            [w2_scales],
            num_experts,
            num_topk,
            hidden,
            local_expert_num,
        )

    def run_quant(self, weight_dtype, dispatch_quant_out_dtype) -> bool:
        torch_npu.npu.set_device(self.rank)
        (
            x,
            topk_ids,
            topk_weights,
            l1_weights,
            l2_weights,
            l1_weights_sf,
            l2_weights_sf,
            num_experts,
            num_topk,
            hidden,
            local_expert_num,
        ) = self._make_quant_inputs(weight_dtype)
        bs = x.shape[0]
        sym_buffer = get_symm_buffer_for_mega_moe(
            self.ep_group,
            num_experts=num_experts,
            num_max_tokens_per_rank=0,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=0,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
            max_recv_token_num=bs * self.world_size * min(local_expert_num, num_topk),
        )
        y, expert_token_nums = mega_moe(
            sym_buffer=sym_buffer,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            l1_weights=l1_weights,
            l2_weights=l2_weights,
            l1_weights_sf=l1_weights_sf,
            l2_weights_sf=l2_weights_sf,
        )
        return self.check_output(y, expert_token_nums, x, local_expert_num)

    def run_quant_e5m2(self) -> bool:
        if not hasattr(torch, "float8_e5m2"):
            return True
        return self.run_quant(torch.float8_e5m2, 23)

    def run_quant_e4m3(self) -> bool:
        if not hasattr(torch, "float8_e4m3fn"):
            return True
        return self.run_quant(torch.float8_e4m3fn, 24)


def _worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue, dtype_name: str):
    try:
        op = MegaMoeRunner(rank, world_size, port)
        op.generate_hcom()
        if dtype_name == "e5m2":
            result = op.run_quant_e5m2()
        elif dtype_name == "e4m3":
            result = op.run_quant_e4m3()
        else:
            raise ValueError(f"Unsupported mega_moe dtype case: {dtype_name}")
        q.put((rank, True, result, ""))
    except Exception:
        q.put((rank, False, False, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_mega_moe_case(dtype_name: str):
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p_list = []
    port = 29501 + random.randint(0, 10000)

    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, port, q, dtype_name))
        p.start()
        p_list.append(p)

    results = [q.get() for _ in range(world_size)]

    for p in p_list:
        p.join()

    errors = [msg for msg in results if not msg[1] or not msg[2]]
    assert not errors, errors
    assert all(p.exitcode == 0 for p in p_list)


@torch.inference_mode()
def test_mega_moe_fp8_e5m2():
    _run_mega_moe_case("e5m2")


@torch.inference_mode()
def test_mega_moe_fp8_e4m3():
    _run_mega_moe_case("e4m3")
