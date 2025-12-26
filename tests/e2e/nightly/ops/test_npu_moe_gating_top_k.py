import itertools
import random

import numpy
import torch
from torch_npu.testing.testcase import TestCase, run_tests

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# random.seed(450)
seed = 45
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.npu.manual_seed_all(seed)  # 如果你用 GPU


def softmax_func(x, axis=None):
    # print("before softmax:",x)
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    # print(" x_sum:",x_sum)
    ans = y / x_sum
    # print(" ans:",ans)
    return ans, x_max, x_sum


class TestNpuMoeGatingTopK(TestCase):

    def moe_gating_top_k_numpy(self,
                               x: np.ndarray,
                               k: int,
                               bias: numpy.ndarray,
                               k_group: int = 1,
                               group_count: int = 1,
                               group_select_mode: int = 0,
                               renorm: int = 0,
                               norm_type: int = 0,
                               y2_flag: bool = False,
                               routed_scaling_factor: float = 1.0,
                               eps: float = 1e-20) -> tuple:

        dtype = x.dtype
        if dtype != np.float32:
            x = x.astype(np.float32)
            if bias is not None:
               bias = bias.astype(np.float32)

        x = x.numpy()
        if bias is not None:
            bias = bias.numpy()

        if norm_type == 0:  # softmax
            x, _, _ = softmax_func(x, -1)
        else:
            x = 1 / (1 + numpy.exp(-x))  # sigmoid

        original_x = x
        if bias is not None:
            x = x + bias

        if group_count > 1:
            x = x.reshape(x.shape[0], group_count, -1)
            if group_select_mode == 0:
                group_x = numpy.amax(x, axis=-1)
            else:
                group_x = numpy.partition(x, -2, axis=-1)[...,
                                                          -2:].sum(axis=-1)
            indices = numpy.argsort(
                -group_x, axis=-1,
                kind='stable')[:, :k_group]  # Indices of top-k_group

            mask = numpy.ones((x.shape[0], group_count), dtype=bool)
            mask[numpy.arange(x.shape[0])[:, None], indices] = False
            x = numpy.where(mask[..., None], float('-inf'), x)
            x = x.reshape(x.shape[0], -1)

        _, indices = torch.sort(torch.from_numpy(x),
                                dim=-1,
                                stable=True,
                                descending=True)
        indices = numpy.asarray(indices[:, :k])

        y = numpy.take_along_axis(original_x, indices, axis=1)

        if norm_type == 1 or renorm == 1:
            y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
        y *= routed_scaling_factor
        if y2_flag:
            y2 = original_x
        else:
            y2 = None
        y = torch.tensor(y, dtype=dtype)
        return y, indices.astype(numpy.int32), y2

    def test_npu_moe_gating_topk_multi(self, device="npu"):

        group_select_modes = [0, 1]
        renorms = [1]
        norm_types = [0, 1]
        group_counts = [1, 8]
        k_ranges = [4, 8, 12, 16, 6, 32]

        x_dim0_range = range(1, 17)  # 1~16
        x_dim1_range = [256, 128, 64, 208, 192, 160]

        # 所有参数组合
        param_product = itertools.product(group_select_modes, renorms,
                                          norm_types, group_counts, k_ranges,
                                          x_dim0_range, x_dim1_range)

        for (group_select_mode, renorm, norm_type, group_count, k, dim0,
             dim1) in param_product:

            group_count = 8
            k = 5
            if k > dim1 // group_count:
                continue

            # ---- 构造输入 ----
            x = numpy.random.uniform(-2, 2, (dim0, dim1)).astype(numpy.float32)
            bias = numpy.random.uniform(-2, 2, (dim1, )).astype(numpy.float32)

            x_tensor = torch.tensor(x, dtype=torch.float32)
            bias_tensor = torch.tensor(bias, dtype=torch.float32)
            # bias_tensor = None
            k_group = random.randint(1, group_count)
            out_flag = False
            routed_scaling_factor = 1.0
            eps = 1e-20

            # ---- error debug ----

            # k = 4
            # x = numpy.random.uniform(-2, 2, (1, 256)).astype(numpy.float32)
            # bias = numpy.random.uniform(-2, 2, (256,)).astype(numpy.float32)

            # x_tensor = torch.tensor(x, dtype=torch.float32)
            # bias_tensor = torch.tensor(bias, dtype=torch.float32)

            # k_group = 1

            # ---- numpy 结果 ----
            y, expert_idx, out = self.moe_gating_top_k_numpy(
                x_tensor,
                k,
                bias=bias_tensor,
                k_group=k_group,
                group_count=group_count,
                group_select_mode=group_select_mode,
                renorm=renorm,
                norm_type=norm_type,
                y2_flag=out_flag,
                routed_scaling_factor=routed_scaling_factor,
                eps=eps,
            )

            # ---- npu 结果 ----

            y_npu, expert_idx_npu, out_npu = torch.ops._C_ascend.moe_gating_top_k(
                x_tensor.npu(),
                k,
                kGroup=k_group,
                groupCount=group_count,
                groupSelectMode=group_select_mode,
                renorm=renorm,
                normType=norm_type,
                outFlag=out_flag,
                routedScalingFactor=routed_scaling_factor,
                eps=eps,
                biasOptional=bias_tensor.npu()
                if bias_tensor is not None else None)

            # ---- 输出当前 case 信息 ----
            print(
                f"[Case] x=({dim0},{dim1}), k={k}, "
                f"group_count={group_count}, select_mode={group_select_mode}, "
                f"norm_type={norm_type}, renorm={renorm},"
                f"k_group={k_group}")
            print(y_npu.shape, expert_idx_npu.shape, out_npu.shape)
            print(x_tensor.dtype)
            print(y)
            print(y_npu.cpu())
            print(y - y_npu.cpu())
            # ---- 校验 ----
            self.assertRtolEqual(y, y_npu.cpu())
            self.assertRtolEqual(expert_idx, expert_idx_npu.cpu().numpy())
            print("ok\n")


if __name__ == "__main__":
    run_tests()
