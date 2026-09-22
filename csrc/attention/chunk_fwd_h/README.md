# ChunkFwdH

FLA NPU 新增代码保留 [BSD 3-Clause](../LICENSE.fla-npu) 及文件头指定的 [CANN 许可证](../LICENSE.CANN)。

## vLLM Ascend 集成范围

本仓模板仅编译 FP32 key-wise gate、exp2、FP32 状态、K=V=128，支持 VK/KV 两种状态布局。
通过现有 `chunk_kda_fwd` torch binding 的 V2 分支调用，随 `csrc/build_aclnn.sh` 构建。
不需要安装 FLA Python 包。下文保留上游完整接口说明；未编译的模式不属于本仓支持范围。

来源：`flashserve/flash-linear-attention-npu@81a7346ddb13340e03cdd2e14cbbbe777bafbcdf`。

[设计文档](docs/design.md) | [API 文档](docs/api.md)

## 功能

`ChunkFwdH` 计算 GDN/KDA 的 chunk 间状态和 `v_new`。稳定 Python 入口为：

```python
from fla_npu.ops.ascendc import chunk_fwd_h

h, v_new, final_state = chunk_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    state_v_first=False,
)
```

`g` 与 `gk` 必须且只能提供一个：

- g-only：`k` 是 raw key，shape 为 `[B, HK, T, 128]`，要求 `HV % HK == 0`。
- gk-only：`k` 是 Prepare 阶段已经生成的 `kg`，shape 为 `[B, HV, T, 128]`；本算子不会再次把 `gk` 乘到 `k` 上。

## 计算语义

记 `E(x)=exp(x)`；`use_exp2=true` 时改为 `E(x)=exp2(x)`。对 value head `h` 的每个 chunk `c`：

```text
H_c = cast_BF16(R_c)
Pacc_c = W_c @ H_c
P_c = cast_PType(Pacc_c)  # StateT=BF16 时 PType=BF16，否则为 FP32
V_new_fp32_c = fp32(U_c) - fp32(P_c)
V_new_c = cast_BF16(V_new_fp32_c)

g-only:
    V_new_g_c[i,:] = cast_BF16(E(g_last - g_i) * V_new_fp32_c[i,:])
    D_c = k_raw_c^T @ V_new_g_c
    R_{c+1} = E(g_last) * R_c + D_c

gk-only:
    D_c = kg_c^T @ V_new_c
    R_{c+1}[k,v] = E(gk_last[k]) * R_c[k,v] + D_c[k,v]
```

Stage0 和 Stage2 均使用 BF16 矩阵乘输入、FP32 累加。Stage0 在写出时按 `PType` 转换，
`D_c` 保持 FP32 进入 Stage3。`StateT` 按以下优先级确定：

1. 存在 `initial_state` 时，取 `initial_state.dtype`。
2. 否则输出 `final_state` 时，取 `final_state.dtype`。
3. 否则使用 FP32 rolling state。

最终 chunk 若不输出 `final_state`，只执行 Stage0/Stage1，跳过 Stage2/Stage3，也不生成 Stage2 的 L1 右操作数。

## 输入输出

| 名称 | dtype | Shape/约束 |
| --- | --- | --- |
| `k` | BF16 | g-only `[B,HK,T,128]`；gk-only `[B,HV,T,128]` |
| `w` | BF16 | `[B,HV,T,128]` |
| `u` | BF16 | `[B,HV,T,128]` |
| `g` | BF16/FP32 | g-only 必传，`[B,HV,T]` |
| `gk` | BF16/FP32 | gk-only 必传，`[B,HV,T,128]` |
| `initial_state` | BF16/FP32 | 可空；`[N,HV,K,V]` 或 `[N,HV,V,K]` |
| `h` | BF16 | `[B,HV,C,K,V]` 或 `[B,HV,C,V,K]` |
| `v_new` | BF16 | `[B,HV,T,128]` |
| `final_state` | BF16/FP32 | 可空；shape/layout 与 `initial_state` 相同 |

`state_v_first=false` 时 state 末两维为 `[K,V]`，为 `true` 时为 `[V,K]`。两种布局均由 kernel 原生处理，不依赖外部转置。

当前仅支持 `K=V=128`、`chunk_size=64`、`save_new_value=true`，支持 A2、A3 和 A5。

所有 aclnn tensor descriptor 必须使用非私有 ND storage/view format；输入可以是 ND view，
host 会在需要时连续化，`h/v_new/final_state` 输出必须是连续 ND。Python 入口只接受标准物理
布局并创建连续 ND 输出，不支持将私有 NPU format 伪装成 ND。

## 变长序列

变长模式使用 BNSD 容器且要求 `B=1`。`cu_seqlens` 必须从 0 开始、以 `T` 结束并严格递增。`chunk_indices` 若提供，必须严格等于由 `cu_seqlens` 和 `chunk_size=64` 推导出的 sequence-major `(seq_id, chunk_id)` 序列。

## 内核分工

- AIC 每个 head round 固定四个 round-head 槽；AIV0 处理 round head 0/2，AIV1 处理 1/3，每个 AIV 使用两个 local slot。
- round 进入 chunk 循环前确定 `head -> kh -> kg slot` 映射。Stage2 只加载本 round 实际需要的 `k_raw/kg`，最多四份；不跨 round 保留，跨 round 复用同一 key head 时重新读取。
- A2/A3 的 P/D 从 L0C 经 Fixpipe 写 GM scratch，AIV 再从 GM 搬入 UB；A5 的 P/D 从 L0C 直接写配对 AIV UB。
- Stage1 的 ND 右操作数统一经 UB MTE3 到 GM，再由 AIC MTE2 搬入 L1 NZ，不使用 UB 到 L1 的直搬通路。
- tail chunk 的 W、kg/k_raw 和 right 在有效 ND 数据覆盖前先由 MTE2 清零对应 L1 NZ 槽，Cube 的 M/K 按 16 对齐，未覆盖位置不会参与有效累加。
- A5 每个 ping/pong slot 使用独立 ready/free 事件。A2/A3 的 mode2 按一组
  `AIC + 2*AIV` 做 pair 集合同步，尾 pair 缺少 head 的 AIV 也参与 dummy wait/set。
  某 slot/pair 的 MTE2、Cube/Fixpipe 或 VEC/MTE3 完成后立即发布，不等待下一组。
- 下一 head round 的 kg/H/W 预取必须等待上一 round 的 P/D、L1 right 和所有 AIV MTE3 完成。

## aclnn 接口

```cpp
aclnnStatus aclnnChunkFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```
