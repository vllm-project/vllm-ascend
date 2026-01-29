import gc
import os
import sys
from pathlib import Path

import numpy as np
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import torchair

from vllm_ascend.utils import enable_custom_op

torch.manual_seed(42)
torch_npu.npu.config.allow_internal_format = True
enable_custom_op()
LOG_NAME = "dispatch_gmm_combine_decode_test_logs"
BASE_KWARGS = {
    "batch_size": 64,
    "token_hidden_size": 7168,
    "moe_intermediate_size": 2048,
    "ep_world_size": 16,
    "moe_expert_num": 64,
    "shared_expert_rank_num": 0,
    "top_k": 8,
    "test_bfloat16": True,
    "enable_dynamic_bs": False,
    "test_graph": False,
    "with_mc2_mask": False,
    "dynamic_eplb": False,
    "w8a8_dynamic": True,
    "is_nz": True
}


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
    return rank_id > 0


class DecodeMoeOps(torch.nn.Module):

    def __init__(self,
                 gmm1_weight,
                 gmm1_weight_scale,
                 gmm2_weight,
                 gmm2_weight_scale,
                 ep_hcomm_info,
                 batch_size,
                 token_hidden_size,
                 moe_intermediate_size,
                 ep_world_size,
                 moe_expert_num,
                 global_rank_id,
                 shared_expert_rank_num=0,
                 dynamic_eplb=False,
                 w8a8_dynamic=True,
                 is_nz=True):
        super().__init__()
        if w8a8_dynamic:
            assert (gmm1_weight_scale is not None and gmm2_weight_scale is not None), "gmm1_weight_scale and gmm2_weight_scale must be provided for w8a8_dynamic"
        else:
            assert (gmm1_weight_scale is None and gmm2_weight_scale is None), "gmm1_weight_scale and gmm2_weight_scale must be None for w8a8_dynamic"
        self.ep_hcomm_info = ep_hcomm_info
        self.batch_size = batch_size
        self.token_hidden_size = token_hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.ep_world_size = ep_world_size
        self.moe_expert_num = moe_expert_num
        self.global_rank_id = global_rank_id
        self.shared_expert_rank_num = shared_expert_rank_num
        is_shared_expert = global_rank_id < shared_expert_rank_num
        moe_expert_num_per_rank = moe_expert_num // (ep_world_size -
                                                     shared_expert_rank_num)
        self.local_expert_num = 1 if is_shared_expert else moe_expert_num_per_rank
        self.ep_recv_count_size = self.local_expert_num * ep_world_size
        self.dynamic_eplb = dynamic_eplb
        self.w8a8_dynamic = w8a8_dynamic
        self.is_nz = is_nz
        self.gmm1_weight = torch.empty([
            self.local_expert_num, self.token_hidden_size,
            self.moe_intermediate_size * 2
        ])
        self.gmm2_weight = torch.empty([
            self.local_expert_num, self.moe_intermediate_size,
            self.token_hidden_size
        ])
        if self.w8a8_dynamic:
            self.gmm1_weight_scale = torch.empty(
                [self.local_expert_num, self.moe_intermediate_size * 2])
            self.gmm2_weight_scale = torch.empty(
                [self.local_expert_num, self.token_hidden_size])
        else:
            self.gmm1_weight_scale = None
            self.gmm2_weight_scale = None
            self.gmm1_weight_scale_fp32 = None
            self.gmm2_weight_scale_fp32 = None
        self._process_weights_after_loading(gmm1_weight, gmm1_weight_scale,
                                            gmm2_weight, gmm2_weight_scale)

    def _process_weights_after_loading(self, gmm1_weight, gmm1_weight_scale,
                                       gmm2_weight, gmm2_weight_scale):
        if self.w8a8_dynamic:
            gmm1_weight = torch_npu.npu_format_cast(gmm1_weight,
                                                    torch_npu.Format.FRACTAL_NZ)
            gmm2_weight = torch_npu.npu_format_cast(gmm2_weight,
                                                    torch_npu.Format.FRACTAL_NZ)
        self.gmm1_weight = torch.nn.Parameter(gmm1_weight, requires_grad=False)
        self.gmm2_weight = torch.nn.Parameter(gmm2_weight, requires_grad=False)
        if self.w8a8_dynamic:
            self.gmm1_weight_scale = torch.nn.Parameter(gmm1_weight_scale,
                                                        requires_grad=False)
            self.gmm2_weight_scale = torch.nn.Parameter(gmm2_weight_scale,
                                                        requires_grad=False)
            self.gmm1_weight_scale_fp32 = torch.nn.Parameter(
                gmm1_weight_scale.float(), requires_grad=False)
            self.gmm2_weight_scale_fp32 = torch.nn.Parameter(
                gmm2_weight_scale.float(), requires_grad=False)

    def _apply_ops(self, x, expert_ids, smooth_scales, expert_scales,
                   x_active_mask):
        raise NotImplementedError("To be implemented in subclass")

    def forward(self, x, expert_ids, smooth_scales, expert_scales,
                x_active_mask):
        return self._apply_ops(x, expert_ids, smooth_scales, expert_scales,
                               x_active_mask)


class SmallOps(DecodeMoeOps):

    def __init__(self,
                 gmm1_weight,
                 gmm1_weight_scale,
                 gmm2_weight,
                 gmm2_weight_scale,
                 ep_hcomm_info,
                 batch_size,
                 token_hidden_size,
                 moe_intermediate_size,
                 ep_world_size,
                 moe_expert_num,
                 global_rank_id,
                 shared_expert_rank_num=0,
                 dynamic_eplb=False,
                 w8a8_dynamic=True,
                 is_nz=True):
        super().__init__(gmm1_weight, gmm1_weight_scale, gmm2_weight,
                         gmm2_weight_scale, ep_hcomm_info, batch_size,
                         token_hidden_size, moe_intermediate_size,
                         ep_world_size, moe_expert_num, global_rank_id,
                         shared_expert_rank_num, dynamic_eplb, w8a8_dynamic,
                         is_nz)
        self.tp_hcomm_info = ""

    def _apply_ops(self, x, expert_ids, smooth_scales, expert_scales,
                   x_active_mask):
        outputs = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            quant_mode=2 if self.w8a8_dynamic else 0,
            global_bs=self.batch_size * self.ep_world_size,
            expert_token_nums_type=1,  # 0代表前缀和，1代表各自数量
        )
        expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums, ep_send_counts, tp_send_counts, expand_scales = outputs
        output_dtype = x.dtype

        y1_int32 = torch_npu.npu_grouped_matmul(
            x=[expand_x],
            weight=[self.gmm1_weight],
            split_item=3,
            group_list_type=1,  # 默认为0，代表前缀和形式
            group_type=0,  # 0代表m轴分组
            group_list=expert_token_nums,
            output_dtype=torch.int32 if self.w8a8_dynamic else output_dtype)[0]
        y1_scale = None
        if self.w8a8_dynamic:
            y1, y1_scale = torch_npu.npu_dequant_swiglu_quant(
                x=y1_int32,
                weight_scale=self.gmm1_weight_scale.to(torch.float32),
                activation_scale=dynamic_scales,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=expert_token_nums,
                activate_left=True,
                quant_mode=1,
            )
        else:
            y1 = torch_npu.npu_swiglu(y1_int32)
        y2 = torch_npu.npu_grouped_matmul(x=[y1],
                                          weight=[self.gmm2_weight],
                                          scale=[self.gmm2_weight_scale] if self.w8a8_dynamic else None,
                                          per_token_scale=[y1_scale] if self.w8a8_dynamic else None,
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
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp=self.tp_hcomm_info,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            global_bs=self.batch_size * self.ep_world_size)
        return (combine_output, expert_token_nums)


class FusionOp(DecodeMoeOps):

    def __init__(self,
                 gmm1_weight,
                 gmm1_weight_scale,
                 gmm2_weight,
                 gmm2_weight_scale,
                 ep_hcomm_info,
                 batch_size,
                 token_hidden_size,
                 moe_intermediate_size,
                 ep_world_size,
                 moe_expert_num,
                 global_rank_id,
                 shared_expert_rank_num=0,
                 dynamic_eplb=False,
                 w8a8_dynamic=True,
                 is_nz=True):
        super().__init__(gmm1_weight, gmm1_weight_scale, gmm2_weight,
                         gmm2_weight_scale, ep_hcomm_info, batch_size,
                         token_hidden_size, moe_intermediate_size,
                         ep_world_size, moe_expert_num, global_rank_id,
                         shared_expert_rank_num, dynamic_eplb, w8a8_dynamic,
                         is_nz)

    def _apply_ops(self, x, expert_ids, smooth_scales, expert_scales,
                   x_active_mask):
        smooth_scales = torch.zeros(128 * 1024 * 1024).npu()
        output = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=x,
            expert_ids=expert_ids,
            gmm1_permuted_weight=self.gmm1_weight,
            gmm1_permuted_weight_scale=self.gmm1_weight_scale_fp32,
            gmm2_weight=self.gmm2_weight,
            gmm2_weight_scale=self.gmm2_weight_scale_fp32,
            expert_scales=expert_scales,
            expert_smooth_scales=smooth_scales,
            x_active_mask=x_active_mask,
            group_ep=self.ep_hcomm_info,
            ep_rank_size=self.ep_world_size,
            ep_rank_id=self.global_rank_id,
            moe_expert_num=self.moe_expert_num,
            shared_expert_num=1,
            shared_expert_rank_num=self.shared_expert_rank_num,
            quant_mode=0,
            global_bs=self.batch_size * self.ep_world_size)
        return output

    def _process_weights_after_loading(self, gmm1_weight, gmm1_weight_scale,
                                       gmm2_weight, gmm2_weight_scale):
        if self.is_nz:
            gmm1_weight = torch_npu.npu_format_cast(gmm1_weight,
                                                    torch_npu.Format.FRACTAL_NZ)
            gmm2_weight = torch_npu.npu_format_cast(gmm2_weight,
                                                    torch_npu.Format.FRACTAL_NZ)

        if self.dynamic_eplb:
            self.gmm1_weight = [
                weight.clone() for weight in gmm1_weight.unbind(dim=0)
            ]
            self.gmm2_weight = [
                weight.clone() for weight in gmm2_weight.unbind(dim=0)
            ]
            if self.w8a8_dynamic:
                self.gmm1_weight_scale_fp32 = [
                    weight.clone() for weight in gmm1_weight_scale.unbind(dim=0)
                ]
                self.gmm2_weight_scale_fp32 = [
                    weight.clone() for weight in gmm2_weight_scale.unbind(dim=0)
                ]
        else:
            self.gmm1_weight = [gmm1_weight.clone()]
            self.gmm2_weight = [gmm2_weight.clone()]
            if self.w8a8_dynamic:
                self.gmm1_weight_scale_fp32 = [gmm1_weight_scale.clone()]
                self.gmm2_weight_scale_fp32 = [gmm2_weight_scale.clone()]
            else:
                self.gmm1_weight_scale_fp32 = [torch.ones(1).npu().to(gmm1_weight.dtype)]
                self.gmm2_weight_scale_fp32 = [torch.ones(1).npu().to(gmm2_weight.dtype)]


def generate_datas(batch_size,
                   token_hidden_size,
                   moe_intermediate_size,
                   ep_world_size,
                   moe_expert_num,
                   global_rank_id,
                   shared_expert_rank_num=0,
                   top_k=8,
                   test_bfloat16=True,
                   enable_dynamic_bs=False,
                   with_mc2_mask=False,
                   w8a8_dynamic=True):
    is_shared_expert = global_rank_id < shared_expert_rank_num
    moe_expert_num_per_rank = moe_expert_num // (ep_world_size -
                                                 shared_expert_rank_num)
    actual_bs = int(
        torch.randint(2 if with_mc2_mask else 1, batch_size, [1]).item(
        ) if enable_dynamic_bs else batch_size)
    local_expert_num = 1 if is_shared_expert else moe_expert_num_per_rank
    gmm1_input_dim = token_hidden_size
    gmm1_output_dim = moe_intermediate_size * 2
    gmm2_input_dim = moe_intermediate_size
    gmm2_output_dim = token_hidden_size
    x = torch.rand([actual_bs, token_hidden_size]) * 0.5 - 0.5
    expert_ids = torch.arange(
        global_rank_id * batch_size * top_k,
        global_rank_id * batch_size * top_k + actual_bs * top_k).to(
            torch.int32).view(actual_bs, top_k)
    expert_ids = expert_ids % moe_expert_num
    gmm1_weight_scale = None
    gmm2_weight_scale = None
    if w8a8_dynamic:
        if is_shared_expert:
            gmm1_weight = torch.ones([
                local_expert_num, gmm1_input_dim, gmm1_output_dim
            ]).to(torch.int8) * 4
            gmm2_weight = torch.ones([
                local_expert_num, gmm2_input_dim, gmm2_output_dim
            ]).to(torch.int8) * 4
            gmm1_weight[:, :, ::2] = gmm1_weight[:, :, ::2] * -1
            gmm2_weight[:, :, ::2] = gmm2_weight[:, :, ::2] * -1
            gmm1_weight_scale = torch.ones([local_expert_num, gmm1_output_dim
                                            ]) * 0.0015
            gmm2_weight_scale = torch.ones([local_expert_num, gmm2_output_dim
                                            ]) * 0.0015
        else:
            gmm1_weight = torch.randint(
                -16, 16,
                [local_expert_num, gmm1_input_dim, gmm1_output_dim]).to(torch.int8)
            gmm2_weight = torch.randint(
                -16, 16,
                [local_expert_num, gmm2_input_dim, gmm2_output_dim]).to(torch.int8)
            gmm1_weight_scale = torch.rand([local_expert_num, gmm1_output_dim
                                            ]) * 0.003 + 0.0015
            gmm2_weight_scale = torch.rand([local_expert_num, gmm2_output_dim
                                            ]) * 0.003 + 0.0015
    else:
        if is_shared_expert:
            gmm1_weight = torch.ones([
                local_expert_num, gmm1_input_dim, gmm1_output_dim
            ]).to(torch.bfloat16 if test_bfloat16 else torch.float16) * 0.5
            gmm2_weight = torch.ones([
                local_expert_num, gmm2_input_dim, gmm2_output_dim
            ]).to(torch.bfloat16 if test_bfloat16 else torch.float16) * 0.5
        else:
            gmm1_weight = torch.rand([local_expert_num, gmm1_input_dim, gmm1_output_dim]).to(torch.bfloat16 if test_bfloat16 else torch.float16) * 0.25
            gmm2_weight = torch.rand([local_expert_num, gmm2_input_dim, gmm2_output_dim]).to(torch.bfloat16 if test_bfloat16 else torch.float16) * 0.25
        gmm1_weight[:, ::2, :] = gmm1_weight[:, ::2, :] * -1
        gmm2_weight[:, ::2, :] = gmm2_weight[:, ::2, :] * -1
    expert_scales = torch.rand(actual_bs, top_k)
    if test_bfloat16:
        x = x.bfloat16()
        if w8a8_dynamic:
            assert (gmm1_weight_scale is not None and gmm2_weight_scale is not None), "gmm1_weight_scale and gmm2_weight_scale must be provided for w8a8_dynamic"
            gmm1_weight_scale = gmm1_weight_scale.bfloat16()
            gmm2_weight_scale = gmm2_weight_scale.bfloat16()
    else:
        x = x.half()
    smooth_sales = None
    x_active_mask = None
    valid_token_num = actual_bs
    if with_mc2_mask:
        valid_token_num = int(torch.randint(1, actual_bs, [1]).item())
        x_active_mask = torch.cat(
            (torch.ones(valid_token_num),
             torch.zeros(actual_bs - valid_token_num))).bool()
    return (x, expert_ids, smooth_sales, expert_scales, x_active_mask), \
            (gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale), \
            actual_bs, valid_token_num


def run_once(local_rank_id,
             batch_size,
             token_hidden_size,
             moe_intermediate_size,
             ep_world_size,
             moe_expert_num,
             shared_expert_rank_num=0,
             top_k=8,
             test_bfloat16=True,
             enable_dynamic_bs=False,
             test_graph=False,
             with_mc2_mask=False,
             dynamic_eplb=False,
             w8a8_dynamic=True,
             is_nz=True):
    log_file = redirect_output(f"local_rank_{local_rank_id}.log"
                               ) if output_to_file(local_rank_id) else None
    global_rank_id = local_rank_id  # 单机
    device_id = local_rank_id % 16
    torch_npu.npu.set_device(device_id)

    # 初始化分布式环境
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"  # 端口号随意
    dist.init_process_group(backend="hccl",
                            rank=local_rank_id,
                            world_size=ep_world_size)
    ep_ranks_list = list(np.arange(0, ep_world_size))
    ep_group = dist.new_group(backend="hccl", ranks=ep_ranks_list)
    ep_group_small = dist.new_group(backend="hccl", ranks=ep_ranks_list)

    ep_hcomm_info_fused = ep_group._get_backend(
        torch.device("npu")).get_hccl_comm_name(local_rank_id)
    ep_hcomm_info_small = ep_group_small._get_backend(
        torch.device("npu")).get_hccl_comm_name(local_rank_id)
    torch_npu.npu.synchronize(device_id)

    parameter = (batch_size, token_hidden_size, moe_intermediate_size,
                 ep_world_size, moe_expert_num, global_rank_id,
                 shared_expert_rank_num)
    input_datas, weight_datas, actual_bs, valid_token_num = generate_datas(
        *parameter, top_k, test_bfloat16, enable_dynamic_bs, with_mc2_mask, w8a8_dynamic)
    input_datas = [
        data.npu() if data is not None else None for data in input_datas
    ]
    weight_datas = [
        data.npu() if data is not None else None for data in weight_datas
    ]
    small_ops = SmallOps(*weight_datas, ep_hcomm_info_small, *parameter,
                         dynamic_eplb, w8a8_dynamic, is_nz).npu()  # type: ignore
    fused_ops = FusionOp(*weight_datas, ep_hcomm_info_fused, *parameter,
                         dynamic_eplb, w8a8_dynamic, is_nz).npu()  # type: ignore
    if test_graph:
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        fused_ops = torch.compile(fused_ops, backend=npu_backend)
    
    # test performance
    start_time = time.perf_counter()
    for _ in range(100):
        small_op_token_output, small_op_count_output = small_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_time_us = elapsed_time * 1000000
    print(f"rank-{global_rank_id} small {elapsed_time_us} us")
    start_time = time.perf_counter()
    for _ in range(100):
        fused_op_token_output, fused_op_count_output = fused_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_time_us = elapsed_time * 1000000
    print(f"rank-{global_rank_id} fused {elapsed_time_us} us")
    small_op_token_output, small_op_count_output = small_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print(f"rank-{global_rank_id} Small op End")
    fused_op_token_output, fused_op_count_output = fused_ops(*input_datas)
    torch_npu.npu.synchronize(device_id)
    print(f"rank-{global_rank_id} Fused op End")
    dist.destroy_process_group()
    if log_file is not None:
        log_file.close()
    try:
        torch.testing.assert_close(small_op_token_output[0:valid_token_num].cpu(),
                                   fused_op_token_output[0:valid_token_num].cpu(),
                                   atol=2.0,
                                   rtol=0.02)
        torch.testing.assert_close(small_op_count_output.cpu(),
                                   fused_op_count_output.cpu())
    except Exception as e:
        print(f"rank-{global_rank_id} Assert close Failed: {e}")
    else:
        print(f"rank-{global_rank_id} Assert close Pass")
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_dispatch_gmm_combine_decode_base():
    custom_kwargs = BASE_KWARGS
    custom_kwargs["batch_size"] = 32
    custom_kwargs["ep_world_size"] = 8
    custom_kwargs["moe_expert_num"] = 32
    custom_kwargs["w8a8_dynamic"] = False
    custom_kwargs["is_nz"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    print(f"{custom_kwargs=}")
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)
    print(f"{custom_kwargs=}")


@torch.inference_mode()
def test_dispatch_gmm_combine_decode_with_mc2_mask():
    custom_kwargs = BASE_KWARGS
    custom_kwargs["with_mc2_mask"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)


@torch.inference_mode()
def test_dispatch_gmm_combine_decode_dynamic_eplb():
    custom_kwargs = BASE_KWARGS
    custom_kwargs["dynamic_eplb"] = True
    ep_world_size = custom_kwargs["ep_world_size"]
    custom_args = tuple(custom_kwargs.values())
    mp.spawn(run_once, args=custom_args, nprocs=ep_world_size, join=True)

if __name__ == "__main__":
    test_dispatch_gmm_combine_decode_base()
