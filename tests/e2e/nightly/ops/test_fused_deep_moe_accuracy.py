import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
use_graph = False
test_bfloat16 = True
enable_dynamic_bs = False
if use_graph:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

enable_custom_op()

TP = 1
print(
    f"{len(sys.argv)= }, {sys.argv= }\n{use_graph= }, {test_bfloat16= }, {enable_dynamic_bs= }"
)
assert len(
    sys.argv
) == 7, "入参列表：[0]rank_size, [1]share_expert_rank_num, [2]moe_expert_num, [3]bs, [4]name, [5]loop_cnt"
ep_world_size = int(sys.argv[1])
SHARE_RANK_NUM = int(sys.argv[2])
MOE_RANK_NUM = ep_world_size - SHARE_RANK_NUM
MOE_EXPERT_NUM = int(sys.argv[3])
MOE_EXPERT_NUM_PER_RANK = MOE_EXPERT_NUM // MOE_RANK_NUM
RANK_BS = int(sys.argv[4])
LOG_NAME = str(sys.argv[5])
loop_times = int(str(sys.argv[6]))
node_num = 1

SHARE_EXPERT_NUM = SHARE_RANK_NUM
DISPATCH_QUANT = True
H = 7168
K = 8
GMM1_INPUT = H
GMM1_HIDDEN = 4096
GMM2_INPUT = GMM1_HIDDEN // 2
GMM2_HIDDEN = H

global_rank_id = 0
ep_hcomm_info = None
ep_hcomm_info_small = None
commArgs = None
tp_hcomm_info = None
device_id = None


def redirect_output(log_file_path):
    log_path = Path(LOG_NAME) / log_file_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOG_NAME + "/" + log_file_path, "w")
    os.dup2(f.fileno(), sys.stdout.fileno())
    os.dup2(f.fileno(), sys.stderr.fileno())
    return f


def permute_weight(w: torch.Tensor, tile_n):
    *dims, n = w.shape
    order = list(range(len(dims))) + [-2, -3, -1]
    return w.reshape(*dims, 2, n // tile_n,
                     tile_n // 2).permute(order).reshape(*dims,
                                                         n).contiguous()


def output_to_file(rank_id):
    # return True
    return rank_id not in [0, SHARE_RANK_NUM]


class SmallOps(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, gmm1_weight, gmm1_weight_scale,
                gmm2_weight, gmm2_weight_scale, smooth_scales, expert_scales):
        outputs = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info_small,
            ep_world_size=ep_world_size,
            ep_rank_id=global_rank_id,
            moe_expert_num=MOE_EXPERT_NUM,
            group_tp=tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=SHARE_RANK_NUM,
            quant_mode=2 if DISPATCH_QUANT else 0,
            global_bs=RANK_BS * ep_world_size,
            expert_token_nums_type=1,  # 0代表前缀和，1代表各自数量
        )
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_send_counts, tp_send_counts, expand_scales = outputs
        output_dtype = torch.bfloat16 if test_bfloat16 else torch.half

        y1_int32 = torch_npu.npu_grouped_matmul(
            x=[expand_x],
            weight=[gmm1_weight],
            split_item=3,
            group_list_type=1,  # 默认为0，代表前缀和形式
            group_type=0,  # 0代表m轴分组
            group_list=expert_token_nums,
            output_dtype=torch.int32)[0]
        y1, y1_scale = torch_npu.npu_dequant_swiglu_quant(
            x=y1_int32,
            weight_scale=gmm1_weight_scale.to(torch.float32),
            activation_scale=dynamic_scales,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=expert_token_nums,
            activate_left=True,
            quant_mode=1,
        )
        y2 = torch_npu.npu_grouped_matmul(x=[y1],
                                          weight=[gmm2_weight],
                                          scale=[gmm2_weight_scale],
                                          per_token_scale=[y1_scale],
                                          split_item=2,
                                          group_list_type=1,
                                          group_type=0,
                                          group_list=expert_token_nums,
                                          output_dtype=output_dtype)[0]
        combine_output = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=y2,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            group_ep=ep_hcomm_info_small,
            ep_world_size=ep_world_size,
            ep_rank_id=global_rank_id,
            moe_expert_num=MOE_EXPERT_NUM,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp=tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=SHARE_RANK_NUM,
            global_bs=RANK_BS * ep_world_size)
        return combine_output


class FusionOp(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, expert_ids, gmm1_weight, gmm1_weight_scale,
                gmm2_weight, gmm2_weight_scale, smooth_scales, expert_scales):
        output = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x, expert_ids, gmm1_weight, gmm1_weight_scale, gmm2_weight,
            gmm2_weight_scale, smooth_scales, expert_scales, ep_hcomm_info,
            ep_world_size, global_rank_id, MOE_EXPERT_NUM, 1, SHARE_RANK_NUM,
            0, RANK_BS * ep_world_size)
        return output


def generate_datas():
    if enable_dynamic_bs:
        actual_bs = torch.randint(1, RANK_BS, [1]).item()
        print(f"rank-{global_rank_id}: {actual_bs=}")
    else:
        actual_bs = RANK_BS
    local_expert_num = 1 if global_rank_id < SHARE_RANK_NUM else MOE_EXPERT_NUM_PER_RANK
    x = torch.rand([actual_bs, H]).half()
    x = x * 10 - 5
    expert_ids = [
        i % MOE_EXPERT_NUM
        for i in range(global_rank_id * RANK_BS *
                       K, global_rank_id * RANK_BS * K + actual_bs * K)
    ]
    expert_ids = torch.Tensor(expert_ids).to(torch.int32).view(actual_bs, K)
    if global_rank_id < SHARE_RANK_NUM:
        gmm1_weight = torch.ones([local_expert_num, GMM1_INPUT, GMM1_HIDDEN
                                  ]).to(torch.int8) * 4
        gmm2_weight = torch.ones([local_expert_num, GMM2_INPUT, GMM2_HIDDEN
                                  ]).to(torch.int8) * 4
        gmm1_weight[:, :, ::2] = gmm1_weight[:, :, ::2] * -1
        gmm2_weight[:, :, ::2] = gmm2_weight[:, :, ::2] * -1
        gmm1_weight_scale = torch.ones([local_expert_num, GMM1_HIDDEN
                                        ]) * 0.0015
        gmm2_weight_scale = torch.ones([local_expert_num, GMM2_HIDDEN
                                        ]) * 0.0015
    else:
        gmm1_weight = torch.randint(
            -16, 16,
            [local_expert_num, GMM1_INPUT, GMM1_HIDDEN]).to(torch.int8)
        gmm2_weight = torch.randint(
            -16, 16,
            [local_expert_num, GMM2_INPUT, GMM2_HIDDEN]).to(torch.int8)
        gmm1_weight_scale = torch.rand([local_expert_num, GMM1_HIDDEN
                                        ]) * 0.003 + 0.0015
        gmm2_weight_scale = torch.rand([local_expert_num, GMM2_HIDDEN
                                        ]) * 0.003 + 0.0015
    expert_scales = torch.rand(actual_bs, K)
    if test_bfloat16:
        x = x.bfloat16()
        gmm1_weight_scale = gmm1_weight_scale.bfloat16()
        gmm2_weight_scale = gmm2_weight_scale.bfloat16()
    else:
        x = x.half()
    return x, expert_ids, gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, None, expert_scales


def test_small_op(x, expert_ids, gmm1_weight, gmm1_weight_scale, gmm2_weight,
                  gmm2_weight_scale, smooth_sales, expert_scales):
    small_op = SmallOps().npu()
    gmm1_weight = torch_npu.npu_format_cast(gmm1_weight, 29)
    gmm2_weight = torch_npu.npu_format_cast(gmm2_weight, 29)
    for _ in range(1, loop_times + 1):
        output = small_op(x, expert_ids, gmm1_weight, gmm1_weight_scale,
                          gmm2_weight, gmm2_weight_scale, smooth_sales,
                          expert_scales)
    return output


def test_fused_op(x, expert_ids, gmm1_weight, gmm1_weight_scale, gmm2_weight,
                  gmm2_weight_scale, smooth_sales, expert_scales):
    fused_op = FusionOp().npu()
    if use_graph:
        fused_op = torch.compile(fused_op, backend=npu_backend)
    gmm1_weight = gmm1_weight.transpose(1,2).contiguous()\
                    .view(-1, 2, 32, 64, 7168).transpose(1,2).contiguous()\
                    .view(-1, 4096, 7168).transpose(1,2).contiguous()
    gmm1_weight = torch_npu.npu_format_cast(gmm1_weight, 2)
    gmm1_weight.add_(0)
    gmm1_weight = torch_npu.npu_format_cast(gmm1_weight, 29)

    gmm1_weight_scale = permute_weight(gmm1_weight_scale, 128)
    gmm2_weight = torch_npu.npu_format_cast(
        gmm2_weight.transpose(1, 2).contiguous(), 29)

    if test_bfloat16:
        gmm1_weight_scale = gmm1_weight_scale.float()
        gmm2_weight_scale = gmm2_weight_scale.float()

    smooth_sales = torch.ones(
        [RANK_BS]).float().npu() if smooth_sales is None else smooth_sales
    for _ in range(1, loop_times + 1):
        # print(f"iter: {_} / {loop_times}")
        output = fused_op(x, expert_ids, gmm1_weight, gmm1_weight_scale,
                          gmm2_weight, gmm2_weight_scale, smooth_sales,
                          expert_scales)
    torch_npu.npu.synchronize(device_id)
    print("fused op run end")
    # 只返回一个出参
    return output[0]


def test():
    tensor_datas = [
        data.npu() if data is not None else None for data in generate_datas()
    ]

    small_op_datas = [
        data.clone().detach() if data is not None else None
        for data in tensor_datas
    ]
    small_op_output = test_small_op(*small_op_datas)
    print(
        f"{small_op_output= }\n {small_op_output.abs().mean()=}, {small_op_output.abs().max()=}"
    )

    fused_op_datas = [
        data.clone().detach() if data is not None else None
        for data in tensor_datas
    ]
    fused_op_output = test_fused_op(*fused_op_datas)
    print(
        f"{fused_op_output= }\n {fused_op_output.abs().mean()=}, {fused_op_output.abs().max()=}"
    )

    diff = (small_op_output - fused_op_output).abs()
    print(
        f"[info-{global_rank_id}] dispatch gmm combine decode: {diff.max()= }, {diff.mean()= }"
    )


def worker(rank, ep_world_size):
    if output_to_file(rank):
        log_file = redirect_output(f"log_test_accuracy_rank_{rank}.txt")
    global global_rank_id, ep_hcomm_info, ep_hcomm_info_small, tp_hcomm_info, device_id
    global_rank_id = rank
    device_id = rank % 16
    torch_npu.npu.set_device(device_id)

    # 初始化分布式环境
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"  # 端口号随意
    dist.init_process_group(backend="hccl",
                            rank=rank,
                            world_size=ep_world_size)

    print(f"[info-{rank}] start ep comm init...")
    ep_ranks_list = list(np.arange(0, ep_world_size))
    print(f"[info-{rank}] ep rank list:", ep_ranks_list)
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    ep_group_small = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    tp_group = dist.new_group(backend="hccl", ranks=[rank])

    ep_hcomm_info = ep_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(rank)
    ep_hcomm_info_small = ep_group_small._get_backend(
        torch.device("npu")).get_hccl_comm_name(rank)
    tp_hcomm_info = tp_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(rank)

    torch_npu.npu.synchronize(device_id)
    print(
        f"[info-{rank}] ep group: {ep_group}, ep_hcomm_info:{type(ep_hcomm_info)}"
    )

    # 测试
    test()
    # 关闭进程组
    torch_npu.npu.synchronize(device_id)
    dist.destroy_process_group()
    if output_to_file(rank):
        log_file.close()


if __name__ == "__main__":
    mp.spawn(worker, args=(ep_world_size, ), nprocs=ep_world_size, join=True)
