import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# The op internally quantizes activations to int8 per token (scale = rowmax/127)
# before each GEMM, so an fp32 reference on the unquantized inputs matches the op
# output within roughly 1% relative error. The K-stride corruption this test
# guards against (#16561) manifests as O(1) relative error instead.
REL_ERR_TOLERANCE = 3e-2

# Global expert id g in [0, world_size * e) is owned by rank g // e; every rank
# builds identical expert weights from the shared seed, so the reference below
# can resolve g through the local weight table (weights[g % e]).
SEED = 1234


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

    @staticmethod
    def torch_reference(x, weight1, weight2, expert_idx, probs):
        """fp32 reference: GMM1 -> SwiGLU(silu(gate) * up, gate = first half) -> GMM2,
        summed over topk with the raw (unnormalized) probs weights."""
        m, _ = x.shape
        _, _, n = weight1.shape
        k2 = n // 2
        topk = expert_idx.shape[1]
        x = x.float()
        w1 = weight1.float()
        w2 = weight2.float()
        out = torch.zeros(x.shape[0], w2.shape[2], dtype=torch.float32)
        for i in range(m):
            acc = torch.zeros(w2.shape[2], dtype=torch.float32)
            for t in range(topk):
                g = int(expert_idx[i, t]) % w1.shape[0]
                y1 = x[i] @ w1[g]
                gate, up = y1[:k2], y1[k2:]
                h = gate * torch.sigmoid(gate) * up
                acc += probs[i, t].item() * (h @ w2[g])
            out[i] = acc
        return out

    def build_case(self, m, k, n, topk, e):
        """Deterministic inputs, identical on every rank (see SEED)."""
        torch.manual_seed(SEED)
        k2 = n // 2
        n2 = k
        x = (torch.randn(m, k, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        # Keep int8 weights small so the int16 C workspace never overflows for
        # the K=2048 case (per-token int8 x spans +/-127 by construction).
        weight1 = torch.randint(-2, 3, (e, k, n), dtype=torch.int8)
        weight2 = torch.randint(-2, 3, (e, k2, n2), dtype=torch.int8)
        expert_idx = torch.randint(0, self.world_size * e, (m, topk), dtype=torch.int32)
        # Routing weights in serving are softmax outputs (non-negative); the op's
        # routing-quant stage (moeInitRoutingQuantV2) is built for that regime.
        probs = torch.softmax(torch.randn(m, topk, dtype=torch.float32), dim=-1)
        return x, weight1, weight2, expert_idx, probs

    def run_case(self, m, k, n, topk, e, tensor_list: bool) -> bool:
        torch_npu.npu.set_device(self.rank)
        torch.manual_seed(SEED)  # reset so build_case output is rank-independent
        x, weight1, weight2, expert_idx, probs = self.build_case(m, k, n, topk, e)
        n2 = k

        torch_npu.npu.config.allow_internal_format = True
        # Scales must be encoded as serving does (w8a8_dynamic.py
        # scale_from_float_to_int64): the fp32 bit pattern widened to int64.
        # The op interprets the bits as a float, so an int64 value of 1 decodes
        # as a ~1e-45 denormal and the output flushes to zero in bf16 -- which
        # is why the old test's randint(0, 1) scales (float 0.0) never showed
        # any output at all. A scale value of 1.0f keeps the fp32 reference
        # free of scale handling.
        scale1 = torch.ones(e, n, dtype=torch.float32).view(torch.int32).to(torch.int64)
        scale2 = torch.ones(e, n2, dtype=torch.float32).view(torch.int32).to(torch.int64)

        if tensor_list:
            weight1_nz_npu = [torch_npu.npu_format_cast(weight1[i].npu(), 29) for i in range(e)]
            scale1_npu = [scale1[i].npu() for i in range(e)]
            weight2_nz_npu = [torch_npu.npu_format_cast(weight2[i].npu(), 29) for i in range(e)]
            scale2_npu = [scale2[i].npu() for i in range(e)]
        else:
            weight1_nz_npu = [torch_npu.npu_format_cast(weight1.npu(), 29)]
            scale1_npu = [scale1.npu()]
            weight2_nz_npu = [torch_npu.npu_format_cast(weight2.npu(), 29)]
            scale2_npu = [scale2.npu()]

        # True = active token (kernel: !isActive -> expert id overwritten with
        # the null expert), and the tiling requires dim0 == M exactly.
        xactmask = torch.ones(m, dtype=torch.bool).npu()
        out = torch.zeros(m, k, dtype=torch.bfloat16).npu()
        # 1-D per-local-expert counter, as in serving (moe_comm_method.py).
        expert_token_nums = torch.zeros(e, dtype=torch.int32).npu()

        print(f"[rank {self.rank}] case m={m} k={k} n={n} tensor_list={tensor_list}: calling op...", flush=True)
        torch.ops._C_ascend.dispatch_ffn_combine(
            x=x.npu(),
            weight1=weight1_nz_npu,
            weight2=weight2_nz_npu,
            expert_idx=expert_idx.npu(),
            bias1=[],
            bias2=[],
            scale1=scale1_npu,
            scale2=scale2_npu,
            probs=probs.npu(),
            group=self.hcomm_info,
            # Must exceed m * topk with headroom: dispatch beyond capacity is
            # silently truncated (see #17343), which zeroes the output.
            max_output_size=4096,
            # Serving passes 0.0 (clamp disabled) for plain SwiGLU models.
            swiglu_limit=0.0,
            world_size=self.world_size,
            x_active_mask=xactmask,
            out=out,
            expert_token_nums=expert_token_nums,
        )
        torch.npu.synchronize()
        print(f"[rank {self.rank}] case m={m} k={k} n={n} tensor_list={tensor_list}: op returned", flush=True)

        ref = self.torch_reference(x, weight1, weight2, expert_idx, probs)
        out_npu = out.float().cpu()
        denom = ref.norm()
        assert denom > 0, "reference output is all-zero; test inputs are degenerate"
        rel = (out_npu - ref).norm() / denom
        assert rel < REL_ERR_TOLERANCE, (
            f"dispatch_ffn_combine output deviates from reference: rel={rel:.3e} "
            f"(tolerance {REL_ERR_TOLERANCE}), shape m={m} k={k} n={n} tensor_list={tensor_list}, "
            f"out_absmax={out_npu.abs().max().item():.3e}, "
            f"out_nonzero_ratio={(out_npu != 0).float().mean().item():.3f}, "
            f"ref_absmax={ref.abs().max().item():.3e}"
        )
        print(f"[rank {self.rank}] case m={m} k={k} n={n} tensor_list={tensor_list}: rel={rel:.3e} OK", flush=True)
        return True

    def run_tensor_list(self) -> bool:
        # Shapes follow the op-level recipe from #16561 (M=128, E=40, topk=7,
        # verified there to produce non-zero op output).
        # N == K: regression guard, must stay correct.
        ok1 = self.run_case(m=128, k=1024, n=1024, topk=7, e=40, tensor_list=True)
        # N > K (DeepSeek-V2-Lite-like MoE geometry: 2*moe_intermediate > hidden)
        ok2 = self.run_case(m=128, k=2048, n=3072, topk=7, e=40, tensor_list=True)
        return ok1 and ok2

    def run_normal(self) -> bool:
        ok1 = self.run_case(m=128, k=1024, n=1024, topk=7, e=40, tensor_list=False)
        ok2 = self.run_case(m=128, k=2048, n=3072, topk=7, e=40, tensor_list=False)
        return ok1 and ok2


def worker(rank: int, world_size: int, port: int, q: mp.SimpleQueue):
    print(f"[worker rank {rank}] started", flush=True)
    op = TestDispatchFFNCombine(rank, world_size, port)
    op.generate_hcom()
    print(f"[worker rank {rank}] hccl ready", flush=True)
    out1 = op.run_tensor_list()
    q.put(("tensor_list", out1))
    out2 = op.run_normal()
    q.put(("normal", out2))


@torch.inference_mode()
def test_dispatch_ffn_combine_kernel():
    world_size = 2
    # torch_npu forbids device init in forked subprocesses once the parent has
    # initialized NPU (enable_custom_op() at import time does); spawn re-imports
    # this module in each worker, which re-registers the custom ops there.
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
