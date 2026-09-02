"""FFN 融合算子(torch.ops._C_ascend.ffn_linear)测试套件。

布局语义:非方阵按 w1 形状识别;全方阵(H==K==N)歧义默认 linear([N,K],PyTorch 惯例)。

测试类型:
  - 精度(NPU):golden 为 bf16 小算子链 + fp32 真值双口径
  - 布局守卫(NPU):canonical/linear/方阵/swiglu 布局校验
  - meta 契约(CPU):device="meta" 验证 shape 推导与 kernel 一致(torch.compile/ACLGraph 依赖)
  - 确定性(NPU):重复运行 bitwise 一致
  - 性能 smoke(NPU):FFN 耗时不显著劣于链(宽松阈值,防大回退)
"""

import gc
import random

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# (act, M, K, H, N):model-typical 大 shape + boundary(奇数尾块/极小 M/大 K)
CASES = [
    # model-typical shapes
    ("gelu", 11, 1024, 4096, 1024),
    ("gelu", 62, 1024, 4096, 1024),
    ("silu", 2, 256, 1024, 1024),
    ("swiglu", 1, 2048, 768, 2048),
    ("swiglu", 1, 2048, 6912, 2048),
    # large-tile shapes
    ("gelu", 540, 1280, 5120, 1280),
    ("swiglu", 640, 896, 4864, 896),
    # boundary: odd-M tail / tiny hidden / large K
    ("gelu", 17, 127, 257, 129),
    ("silu", 33, 63, 129, 65),
    ("silu", 1024, 8192, 2048, 1024),
]


def _gen(act, m, k, h, n, dev):
    rows = 2 * h if act == "swiglu" else h
    torch.manual_seed(0)
    x = (torch.randn(m, k, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    # Linear 布局:weight1 [rows,K](swiglu 为 [2H,K]),weight2 [N,H]
    w1 = (torch.randn(rows, k, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(n, h, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    if act == "swiglu":
        return x, w1, w2, None, None
    b1 = (torch.randn(rows, dtype=torch.float32, device=dev) * 0.01).to(torch.bfloat16)
    b2 = (torch.randn(n, dtype=torch.float32, device=dev) * 0.01).to(torch.bfloat16)
    return x, w1, w2, b1, b2


def _chain(act, x, w1, w2, b1, b2):
    """golden:官方 bf16 小算子链(linear + act + linear)。"""
    import torch.nn.functional as F

    up = F.linear(x, w1, b1)
    if act == "swiglu":
        hidden = torch_npu.npu_swiglu(up, dim=-1)
    elif act == "silu":
        hidden = F.silu(up)
    else:
        hidden = F.gelu(up)
    return F.linear(hidden, w2, b2)


def _golden_fp32(act, x, w1, w2, b1, b2):
    """golden:fp32 真值(镜像 kernel 语义:fp32 累加域激活,末端 bf16)。"""
    import torch.nn.functional as F

    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    b1f = None if b1 is None else b1.float()
    b2f = None if b2 is None else b2.float()
    up = F.linear(xf, w1f, b1f)
    if act == "swiglu":
        gate, up_half = up.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up_half
    elif act == "silu":
        hidden = F.silu(up)
    else:
        hidden = F.gelu(up)
    return F.linear(hidden, w2f, b2f)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_ffn_linear_precision():
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "ffn_linear"), "vllm_ascend_C 未注册 ffn_linear(检查构建)"
    dev = torch.device("npu")
    for act, m, k, h, n in CASES:
        x, w1, w2, b1, b2 = _gen(act, m, k, h, n, dev)
        y = torch.ops._C_ascend.ffn_linear(x, w1, w2, b1, b2, act, 0)
        torch.npu.synchronize()
        y_chain = _chain(act, x, w1, w2, b1, b2)
        y_fp32 = _golden_fp32(act, x, w1, w2, b1, b2)
        torch.npu.synchronize()

        # 口径 1:不劣于官方链(相对 fp32 真值的误差比 ≤ 1.5,涵盖 bf16 舍入水平差)
        err_ffn = (y.float() - y_fp32).abs().max().item()
        err_chain = (y_chain.float() - y_fp32).abs().max().item()
        assert err_ffn <= max(err_chain * 1.5, 2e-2), (
            f"[{act} {m}x{k}x{h}x{n}] ffn err {err_ffn} vs chain err {err_chain}"
        )
        # 口径 2:shape
        exp_last = n
        assert y.shape == (m, exp_last), f"shape {tuple(y.shape)} != {(m, exp_last)}"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
def test_ffn_linear_layout_guard():
    """布局守卫:canonical [K,N] gelu 可用;swiglu 仅支持 linear [2H,K];不匹配维度报错。"""
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "ffn_linear")
    dev = torch.device("npu")
    x = torch.randn(4, 64, device=dev).to(torch.bfloat16)
    w1n = torch.randn(128, 64, device=dev).to(torch.bfloat16)  # linear [N,K]
    w1c = torch.randn(64, 128, device=dev).to(torch.bfloat16)  # canonical [K,N]
    w2 = torch.randn(32, 128, device=dev).to(torch.bfloat16)
    b = torch.randn(128, device=dev).to(torch.bfloat16)

    y = torch.ops._C_ascend.ffn_linear(x, w1n, w2, b, None, "gelu", 0)
    assert y.shape == (4, 32)
    y = torch.ops._C_ascend.ffn_linear(x, w1c, w2, None, None, "gelu", 0)
    assert y.shape == (4, 32)
    # 全方阵（w1、w2 与 xK 同边长）真歧义:默认 linear([N,K],F.linear 语义)
    w1sq = torch.randn(64, 64, device=dev).to(torch.bfloat16)
    w2sq = torch.randn(32, 64, device=dev).to(torch.bfloat16)
    y = torch.ops._C_ascend.ffn_linear(x, w1sq, w2sq, None, None, "gelu", 0)
    yref = torch.nn.functional.linear(x, w1sq) @ w2sq.T
    torch.npu.synchronize()
    assert y.shape == (4, 32)
    assert (y.float() - yref.float()).abs().max() < 0.05, "square default-linear mismatch"

    with pytest.raises(RuntimeError):
        # swiglu + 纯 canonical 布局(w1 非方阵 [K,2H])必须报错
        w2s = torch.randn(32, 64, device=dev).to(torch.bfloat16)
        torch.ops._C_ascend.ffn_linear(x, w1c, w2s, None, None, "swiglu", 0)


# ---------------------------------------------------------------------------
# Meta 契约测试(CPU 可跑):验证 torch_binding_meta.cpp 的 shape 推导与
# kernel 布局规则一致。device="meta" tensor 零内存,只传播 shape/dtype,
# 是 torch.compile / ACLGraph capture 的依赖路径(参照 #13954 模式)。
# ---------------------------------------------------------------------------


def _meta(x_shape, w1_shape, w2_shape, act):
    """在 meta device 上调用 ffn_linear,返回输出 tensor。"""
    dt = torch.bfloat16
    x = torch.empty(*x_shape, device="meta", dtype=dt)
    w1 = torch.empty(*w1_shape, device="meta", dtype=dt)
    w2 = torch.empty(*w2_shape, device="meta", dtype=dt)
    return torch.ops._C_ascend.ffn_linear(x, w1, w2, None, None, act, 0)


@pytest.mark.parametrize("act", ["gelu", "silu", "swiglu"])
def test_ffn_linear_meta_contract(act):
    """meta kernel shape 推导:Linear/canonical/方阵/swiglu 布局全覆盖。"""
    enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "ffn_linear")
    # x=[4,64], H=128, N=32
    if act == "swiglu":
        w1 = (256, 64)  # [2H,K]
        w2 = (32, 128)  # [N,H]
    else:
        w1 = (128, 64)  # [H,K] (Linear) 或 [K,H] (canonical)
        w2 = (32, 128)  # [N,H] (Linear) 或 [H,N] (canonical)

    # 1. Linear 布局:w1=[N_in,K],输出 last_dim = w2.dim0
    y = _meta((4, 64), w1, w2, act)
    assert y.shape == (4, 32), f"Linear layout: {y.shape} != (4,32)"
    assert y.dtype == torch.bfloat16

    # 2. canonical 布局(gelu/silu):w1=[K,H],输出 last_dim = w2.dim1
    if act != "swiglu":
        w1c = (64, 128)  # [K,H]
        w2c = (128, 32)  # [H,N]
        y = _meta((4, 64), w1c, w2c, act)
        assert y.shape == (4, 32), f"canonical layout: {y.shape} != (4,32)"

    # 3. 方阵歧义:全方阵默认 Linear(gelu/silu);swiglu canonical 解释应报错
    w1sq = (64, 64)
    w2sq = (32, 64)
    if act == "swiglu":
        with pytest.raises(RuntimeError, match="swiglu only supports linear"):
            _meta((4, 64), w1sq, w2sq, act)
    else:
        y = _meta((4, 64), w1sq, w2sq, act)
        assert y.shape == (4, 32), f"square layout: {y.shape} != (4,32)"

    # 4. 3D 输入(batch):x=[2,4,64],输出=[2,4,32]
    y = _meta((2, 4, 64), w1, w2, act)
    assert y.shape == (2, 4, 32), f"batch dim: {y.shape} != (2,4,32)"


# ---------------------------------------------------------------------------
# 确定性测试(NPU):同一输入重复运行,输出必须 bitwise 一致。
# ---------------------------------------------------------------------------

DETERMINISM_REPEATS = 10
DETERMINISM_SHAPES = [
    ("gelu", 62, 1024, 4096, 1024),
    ("swiglu", 1, 2048, 768, 2048),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.parametrize("act,m,k,h,n", DETERMINISM_SHAPES)
def test_ffn_linear_determinism(act, m, k, h, n):
    """重复运行 bitwise 一致:防止 flag 竞态/tile 分配不确定性。"""
    enable_custom_op()
    dev = torch.device("npu")
    x, w1, w2, b1, b2 = _gen(act, m, k, h, n, dev)
    outputs = []
    for _ in range(DETERMINISM_REPEATS):
        y = torch.ops._C_ascend.ffn_linear(x, w1, w2, b1, b2, act, 0)
        torch.npu.synchronize()
        outputs.append(y.clone())
    ref = outputs[0]
    for i, y in enumerate(outputs[1:], 1):
        assert torch.equal(ref, y), f"[{act} {m}x{k}x{h}x{n}] repeat {i}: bitwise mismatch"


# ---------------------------------------------------------------------------
# 性能 smoke 测试(NPU):FFN 耗时不显著劣于链(宽松阈值,防大回退)。
# ---------------------------------------------------------------------------


def _is_ascend_950():
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU unavailable")
@pytest.mark.skipif(not _is_ascend_950(), reason="Only validated on Ascend950PR")
@pytest.mark.skip_global_cleanup
def test_ffn_linear_perf_smoke():
    """性能 smoke:FFN 中位耗时 ≤ 链中位耗时 × 1.5(宽松,只防大回退)。"""
    import statistics
    import time

    enable_custom_op()
    dev = torch.device("npu")
    act, m, k, h, n = ("gelu", 62, 1024, 4096, 1024)
    x, w1, w2, b1, b2 = _gen(act, m, k, h, n, dev)

    def _time_op(fn, warmup=10, reps=5, batch=50):
        # 批量测量:每 batch 次调用做一次 sync,摊薄 Python dispatch / kernel launch
        # 开销(单次 sync 的 wall-clock 会被 dispatch 路径差淹没,不反映算子性能)
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(batch):
                fn()
            torch.npu.synchronize()
            times.append((time.perf_counter() - t0) / batch)
        return statistics.median(times)

    t_ffn = _time_op(lambda: torch.ops._C_ascend.ffn_linear(x, w1, w2, b1, b2, act, 0))
    t_chain = _time_op(lambda: _chain(act, x, w1, w2, b1, b2))
    assert t_ffn <= t_chain, f"perf regression: FFN {t_ffn * 1e6:.1f}µs > chain {t_chain * 1e6:.1f}µs"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
