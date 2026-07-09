import os

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
from vllm_ascend.ops.mega_moe import npu_get_mega_moe_ccl_buffer_size as get_mega_moe_ccl_buffer_size

E = 4
BS = 256
H = 4096
N = 1024
topK = 6
num_experts = 8

server_num = 1
rank_per_dev = 2
world_size = server_num * rank_per_dev
ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
server_index = 0


def ceil(a, b):
    return (a + b - 1) // b

def random_packed_float4(shape):
    if shape[-1] % 2 != 0:
        raise ValueError(f"float4 packed last dim must be even, got shape {shape}")
    unpacked = torch.randint(0, 16, shape, dtype=torch.uint8)
    return unpacked[..., 0::2] | (unpacked[..., 1::2] << 4)

def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    print(f"current device set: {torch_npu.npu.current_device()}")

def init_hccl_comm(rank):
    # 创建HCCL通信链路并初始化EP域
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    master_ip = '127.0.0.1'
    master_port = os.environ.get('MASTER_PORT', '50001')
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{master_ip}:{master_port}',
    )
    print(f"device_{rank} init_process_group success")

    print(f"device {rank} 初始化EP域")
    for ep_ranks in ep_ranks_list:
        tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group = tmp_group

    # Initialize the EP communicator collectively before CommContextManager
    # acquires its HCCL channel. Querying the name on each rank directly can
    # let one rank enter channel creation while its peer is still initializing.
    dist.barrier(group=ep_group)
    ep_backend = ep_group._get_backend(torch.device("npu"))
    try:
        ep_hcomm_info = ep_backend.get_hccl_comm_name(rank, init_comm=False)
    except TypeError:
        ep_hcomm_info = ep_backend.get_hccl_comm_name(rank)

    return ep_hcomm_info, ep_group

def get_megamoe_kwargs(
    x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
):
    x = x.to(torch.bfloat16).npu()
    expert_ids = expert_ids.to(torch.int32).npu()
    weights1 = weights1.to(torch.uint8).npu()
    weights1 = torch_npu.npu_format_cast(
        weights1,
        29,
    )
    weights_scales1 = weights_scales1.to(torch.float8_e8m0fnu).npu()
    weights2 = weights2.to(torch.uint8).npu()
    weights2 = torch_npu.npu_format_cast(
        weights2,
        29,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    )
    weights_scales2 = weights_scales2.to(torch.float8_e8m0fnu).npu()
    expert_scales = expert_scales.to(torch.bfloat16).npu()

    return {
        'x': x,
        'topk_ids': expert_ids,
        'topk_weights': expert_scales,
        'l1_weights': [weights1],
        'l1_weights_sf': [weights_scales1],
        'l2_weights': [weights2],
        'l2_weights_sf': [weights_scales2],
        'weight1_type': 296,
        'weight2_type': 296,
    }

def run_megamoe_npu(
    queue, rank, x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
):
    print(f"{os.getpid()=}{rank=}")
    set_device(rank)
    # HCCL_BUFFSIZE is consumed while the communicator is initialized, so it
    # must be set before init_hccl_comm rather than before SymmBuffer creation.
    buffer_size = get_mega_moe_ccl_buffer_size(
        world_size, num_experts, BS, topK, H,
        dispatch_quant_mode=4, dispatch_quant_out_dtype=296,
    )
    os.environ['HCCL_BUFFSIZE'] = f'{buffer_size}'
    print(f"[INFO] device_{rank} buffer_size is {buffer_size}")
    ep_hcomm_info, ep_group = init_hccl_comm(rank)
    print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
    megamoe_kwargs = get_megamoe_kwargs(
        x=x,
        expert_ids=expert_ids,
        weights1=weights1,
        weights_scales1=weights_scales1,
        weights2=weights2,
        weights_scales2=weights_scales2,
        expert_scales=expert_scales,
    )
    # 构造distribute_buffer（SymmBuffer结构体）
    distribute_buffer = get_symm_buffer_for_mega_moe(
        ep_group, num_experts=num_experts,
        num_max_tokens_per_rank=0, num_topk=topK,
        hidden=H, intermediate_hidden=0,
        dispatch_quant_mode=4, dispatch_quant_out_dtype=296,
        max_recv_token_num=x.shape[0] * world_size * min(E, topK),
    )
    print(f"[INFO] 运行mega_moe")
    # 步骤3：运行mega_moe，传入上一步构造的sym_buffer
    y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

    torch.npu.synchronize()
    print(f"[INFO] device_{rank} finish\n")
    dist.destroy_process_group()
    print(f'rank {rank} epid {rank} npu finished! \n')

    # Sending tensors through torch.multiprocessing moves their storages into
    # shared memory. On an NFS-backed temp directory those live handles become
    # .nfs files and make multiprocessing finalization fail. This smoke test
    # only needs output metadata, so keep the queue payload storage-free.
    queue.put([
        rank,
        [
            {"shape": tuple(y.shape), "dtype": str(y.dtype)},
            {
                "shape": tuple(expert_token_nums.shape),
                "dtype": str(expert_token_nums.dtype),
            },
        ],
    ])

def gen_npu(target_func, **server_kwargs):
    def parse_rank_input(target_func, result_queue, rank, server_kwargs):

        ep_id = rank // 1

        if target_func == run_megamoe_npu:
            return {
                "queue": result_queue,
                "rank": rank,
                "x": server_kwargs["x_list"][ep_id],
                "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                "weights1": server_kwargs["weights1_list"][ep_id],
                "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                "weights2": server_kwargs["weights2_list"][ep_id],
                "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                "expert_scales": server_kwargs["expert_scales_list"][ep_id]
            }


    print("single_server scene!!!!!")
    rank_list = list(range(world_size))
    print(f"rank list is: {rank_list}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    proc_list = []
    for rank in rank_list:
        rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
        proc = ctx.Process(target=target_func, kwargs=rank_kwargs)
        proc.start()
        proc_list.append(proc)


    rank_outputs = [None] * rank_per_dev
    for proc in proc_list:
        rank_id, rank_output = result_queue.get()
        local_rank_id = rank_id - server_index * rank_per_dev
        rank_outputs[local_rank_id] = rank_output


    for proc in proc_list:
        proc.join()

    if None in rank_outputs:
        print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
        exit(1)

    # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
    category_outputs = []
    category_num = len(rank_outputs[0])
    for category_id in range(category_num):
        specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
        category_outputs.append(specific_category_output)

    return category_outputs

if __name__ == "__main__":
    x_shape = [BS, H]
    expert_idx_shape = [BS, topK]
    weight_shape = [E, N, H]
    weight_scale_shape = [E, N, ceil(H, 64), 2]
    output_shape = [BS, N//2]
    weight2_shape = [E, H, N//2]
    weight2_scale_shape = [E, H, ceil(N//2, 64), 2]
    expert_scales_shape = [BS, topK]
    # 构造输入
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    expert_scales = torch.randn(expert_scales_shape, dtype=torch.bfloat16)
    expert_ids = torch.stack(
        [torch.randperm(num_experts)[:topK] for _ in range(BS)]
    ).to(torch.int32)
    weight1 = random_packed_float4(weight_shape)
    
    weight_scales1 = torch.randint(125, 130, weight_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    weight2 = random_packed_float4(weight2_shape)
    
    weight_scales2 = torch.randint(125, 130, weight2_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    golden_x_list = [x.clone() for _ in range(rank_per_dev)]
    golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
    golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
    golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
    golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
    golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
    golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

    [y, expert_token_nums] = gen_npu(
        run_megamoe_npu,
        x_list=golden_x_list,
        expert_ids_list=golden_expert_ids_list,
        weights1_list=golden_weights1_list,
        weights_scales1_list=golden_weights_scales1_list,
        weights2_list=golden_weights2_list,
        weights_scales2_list=golden_weights_scales2_list,
        expert_scales_list=golden_expert_scales_list,
    )
