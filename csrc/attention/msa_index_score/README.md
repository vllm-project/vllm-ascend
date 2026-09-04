# MsaIndexScore

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  |    √     |
| <term>Ascend 950PR/Ascend 950DT</term>                    |    √     |

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）模块 Index Branch 中的 block score。对每个 query token 与每个 KV sparse block，取该 block 内所有因果可见 token 的 $Q_{idx}$ 和 $K_{idx}$（可选 int8 反量化）的"matmul+maxpool"运算，得到逐 block 的重要性分数 `score`，用作 Index Branch 中后续 TopK 的输入。Prefill 与 Decode 由同一接口承载。

- **计算公式**：

    - 非量化场景：

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8 量化场景：

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T} ]
    $$

    完整公式：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    其中 Maxpool 按 sparse block（长度为 $block\_size$）在 KV token 维上取最大值。`start_loc`、`init_blocks`、`local_blocks` 共同生成 $local\_mask$，对序列头部 / 当前 query 附近若干强制保留的 block 写入高分，保证后续 TopK 一定选中它们。与 Triton raw score kernel 对齐时可将 `init_blocks`、`local_blocks` 置 0，关闭 $local\_mask$。

## 参数说明

> **说明：**
>
> - B（Batch Size）表示输入样本批量大小
> - S（Sequence Length）表示序列长度，$S1$ 为 query 侧、$S2$ 为 key 侧
> - T 表示所有 Batch 序列长度累加和，$T1$ 为 query 侧、$T2$ 为 key 侧
> - N（Head Num）表示头数，$N1$ 为 query 侧、$N2$ 为 key 侧
> - D（Head Dim）表示单个注意力头维度
> - PageAttention 场景下 $block\_num$ 为物理 block 总数、$block\_size$ 为每个 block 的 token 数，$maxBlockNumPerSeq$ 为每个 batch 最大逻辑 block 数（通常 $\ge\lceil S2/block\_size\rceil$），$M_b=\lceil S2/block\_size\rceil$ 为逻辑 block 总数

<table style="undefined;table-layout: fixed; width: 1280px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 560px">
<col style="width: 320px">
<col style="width: 120px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的 $Q_{idx}$。当前仅支持 TND，shape 为 $[T1, N1, D]$。</td>
    <td>BFLOAT16、FLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的 $K_{idx}$。支持 TND（$[T2, N2, D]$）、BNBD（$[block\_num, N2, block\_size, D]$）、BBND（$[block\_num, block\_size, N2, D]$）。Ascend 950 上 PA（BBND/BNBD）允许首轴（物理 page）按 stride 非连续存放，page 内其余轴须连续。</td>
    <td>BFLOAT16、FLOAT16、INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>可选输入</td>
    <td>PageAttention 的逻辑 block → 物理 page 映射表。PA 场景必须传入，二维，第二维长度不能小于 $maxBlockNumPerSeq$；shape 为 $[B, S2/block\_size]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>可选输入</td>
    <td>公式中的 $scale$，反量化系数。非量化必须为空；量化场景必选。PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$；TND 为 $[T2, N2]$。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>可选输入</td>
    <td>控制因果可见的 mask。仅在 <code>sparse_mode=3</code> 时使用；取值为 1 表示该位不参与计算，为 0 表示参与计算；shape 为 $[2048, 2048]$。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>可选输入</td>
    <td>每个 Batch 中 Query 的有效 token 数。query 为 TND 时必须传入，单调不减（前缀和），shape 为 $[B+1]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>可选输入</td>
    <td>每个 Batch 中 Key 的有效 token 数。key 为 TND 时必须传入（前缀和）；PageAttention 场景下为各请求可见 $S2$，shape 为 $[B]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>输入</td>
    <td>当前 query 所在逻辑 block 索引（非 token 前缀），用于生成 $local\_mask$；shape 为 $[B]$。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>属性</td>
    <td>key 布局。<code>"TND"</code> / <code>"BBND"</code> / <code>"BNBD"</code>。aclnn 参数名为 <code>layoutKeyOptional</code>，不传时默认 <code>"BBND"</code>。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>属性</td>
    <td>sparse 模式。0：defaultMask（<code>atten_mask</code> 传空）；3：rightDownCausal（须传入 $[2048, 2048]$ 的 <code>atten_mask</code>）。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>属性</td>
    <td>$local\_mask$ 强制选中的头部 block 数。对逻辑 block $[0, init\_blocks)$ 写入高分 $1\mathrm{e}30$。可选，默认 $0$。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>属性</td>
    <td>$local\_mask$ 强制选中的局部窗口长度。窗口为 $[max(0, start\_loc+1-local\_blocks), start\_loc]$，写入高分 $1\mathrm{e}29$（覆盖同位置的 <code>init_blocks</code>）。可选，默认 $1$（对齐 MiniMax HF）；与 Triton raw score 对齐时置 $0$。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>score</td>
    <td>输出</td>
    <td>公式中的 $score$，逐 block 重要性分数；shape 为 $[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 当前 $block\_size$ 仅支持 128。
- `layout_key` 必须显式指定：`"BBND"` / `"BNBD"` / `"TND"`，与 `key` 实际 shape 一致。
- PageAttention（`layout_key` 为 `"BBND"` / `"BNBD"`）场景下，`block_table` 必须传入；TND key 场景不得传入 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和。
- 非量化场景下，`key` dtype 与 `query` 相同（BFLOAT16 / FLOAT16；Ascend 950 另支持 HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN），`scale` 必须为空。量化场景仅支持 INT8（fp16 query），`scale` 必选：PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$，TND 为 $[T2, N2]$，dtype 为 FLOAT。三种 FP8 仅 950，且 query 与 key 必须同型、不得传 `scale`。
- `sparse_mode` 当前仅支持 0、3：
    - 为 0 时，代表 defaultMask 模式，`atten_mask` 传入空；
    - 为 3 时，代表 rightDownCausal 模式，`atten_mask` 必须传入，shape 为 $[2048, 2048]$，取值为 1 代表该位不参与计算，为 0 代表该位参与计算。
- `init_blocks`、`local_blocks` 必须 $\ge 0$ 且不超过逻辑 block 数（PA 为 `block_table` 第二维；TND 为 score 末维对齐宽度）。两者均为 0 时跳过 $local\_mask$。
- A2/A3 与 Ascend 950：`q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 填 `-inf`；整 batch `T1=0` 时 `SetBlockDim(1)`。
- 本算子输出止于 block score，**不包含** TopK。

## 调用示例

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 180px">
<col style="width: 420px">
<col style="width: 380px">
</colgroup>
<thead>
  <tr>
    <th>调用方式</th>
    <th>样例代码</th>
    <th>说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>aclnn 单算子调用</td>
    <td><a href="./examples/test_aclnn_msa_index_score.cpp">test_aclnn_msa_index_score.cpp</a></td>
    <td>内置 CPU golden 的端到端精度自验证</td>
  </tr>
  <tr>
    <td>接口文档</td>
    <td><a href="./docs/aclnnMsaIndexScore.md">aclnnMsaIndexScore.md</a></td>
    <td>两段式接口说明</td>
  </tr>
  <tr>
    <td>测试说明</td>
    <td><a href="./tests/README.md">tests/README.md</a></td>
    <td>用例矩阵与运行方式</td>
  </tr>
</tbody>
</table>

编译与运行（Ascend 950，x86 包名为 `linux-x86_64.run`；A2/A3 将 `--soc` 换成对应平台）：

```bash
cd /path/to/ops-transformer-ar-950
bash build.sh --pkg --soc=ascend950 --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/path/to/msa_opp/
source /path/to/msa_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=/path/to/msa_opp/vendors/custom_transformer

# aclnn example 必须带 --soc=ascend950（否则默认 910b）
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom --soc=ascend950
# 通过：末行 [PASS]: 37/37 cases passed
# 矩阵：30 条 fp16/bf16/int8（BBND/BNBD/TND，含 mixed-batch pad + 整 batch q_len/kv_len=0）+ 3 条 FP8（D=128）+ 4 条 PA key dim0 stride
# 容差：fp16/bf16/int8 1e-3，FP8 2e-2
# A2/A3：同上矩阵去掉 FP8，期望 [PASS]: 34/34（skipped 3 FP8）
```

torch_extension 见 [msa_index_score.md](../../torch_extension/cann_ops_transformer/docs/zh/msa_index_score.md)。`cann/set_env.sh` 会把 cann 自带 `site-packages` 插到 `PYTHONPATH` 前面，必须把本仓 `torch_extension` 再插回最前。

> **实现备注（A2/A3 / Ascend 950）**
>
> - key 布局由属性 `layout_key`（aclnn：`layoutKeyOptional`）指定，支持 PageAttention **BBND** / **BNBD**，以及 packed **TND**（无 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和）。默认 `"BBND"`。
> - `sparse_mode=3` 的 `atten_mask[2048,2048]` 在 host 校验必选；device 侧按 rightDownCausal
>   解析可见窗口（与 LightningIndexer 一致），不逐元素加载模板。
> - `start_loc` 为逻辑 block 索引，与属性 `init_blocks`（默认 0）、`local_blocks`（默认 1）
>   一起在 Maxpool 之后施加 `local_mask`。
> - 完整公式：`score = Maxpool[(scale·)Q@Kᵀ + atten_mask] + local_mask`。
> - **950 当前交付**（`op_kernel/arch35/`）：计算骨架与 A2 相同（Q 驻留 × K pingpong × **8-page S GM** + MODE 0x2 握手）。Cube **原生** FP8（TilingKey 4/5/6，`hifloat8_t` / `fp8_e5m2_t` / `fp8_e4m3fn_t`，无 scale，禁止 Cast→fp16）。L0C→AIV UB（C_to_UB）尚未作为主路径。`arch22/` 冻结。
> - **PA key dim0 stride**（A2/A3 与 950）：key 为 `IgnoreContiguous`；tiling 用 `GetInputStride` / `GetRequiredInputStride` 读首轴元素 stride，写入已有 `strideKvBlock`。TND 不允许非连续。scale 仍按逻辑 page 紧凑布局。
> - torch 的 `torch_npu.hifloat8.npu()` 当前会打出非法 device id，脚本跳过；kernel / aclnn 已挂 HIFLOAT8。
> - 测性能须选 Health=OK 且空闲的卡。
