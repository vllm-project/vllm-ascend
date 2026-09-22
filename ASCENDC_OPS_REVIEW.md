# vllm-ascend AscendC 算子静态分析报告

> 目标：梳理仓库中现有 AscendC 算子、本机（910B3）可开发/可测试的能力，为性能优化选题提供依据。
> 约束：当前自定义算子正在编译中，本报告全部基于**静态分析**（代码/构建脚本/测试目录），运行时结论需后续 profile 验证。

---

## 1. 环境与前提

| 项 | 值 |
|---|---|
| 硬件 | 8 × Ascend 910B3（64GB HBM），`npu-smi` 正常 |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`），SoC 世代 `ascend910b` |
| 框架 | torch 2.10.0 + torch_npu（可导入，识别到 910B3） |
| 仓库 | 分支 `910B_perf`，vllm_ascend 0.19.1rc2.dev 以 editable 方式安装 |
| 进行中 | `csrc/build_out/cann-ops-transformer-custom_linux-aarch64.run` 已产出，`opp/vendors` 下暂只装了 `batch_invariant` —— 主构建/安装尚未完成 |

---

## 2. 算子代码布局与两条构建路径

```
csrc/
├── kernels/                 # ① AscendC kernel（bgmv/sgmv ×4）
├── mla_preprocess/op_kernel/        # ① AscendC kernel
├── batch_matmul_transpose/op_kernel/ # ① AscendC kernel
├── moe/       (18 个子目录)  # ② aclnn 自定义算子（op_host + op_kernel）
├── attention/ (28 个子目录)  # ② 同上
├── gmm/       grouped_matmul_swiglu_quant*  # ② 同上
├── mc2/       dispatch_ffn_combine*        # ② 仅 A3 构建列表
├── torch_binding.cpp / torch_binding_meta.cpp  # 统一注册 torch.ops._C_ascend.*
└── build.sh / build_aclnn.sh                  # ② 的构建/安装脚本
```

**两条构建路径，最终都注册到同一个命名空间 `torch.ops._C_ascend`：**

| | 路径 ①：扩展内置 AscendC | 路径 ②：aclnn 自定义算子包 |
|---|---|---|
| 源码 | `csrc/kernels/*.cpp`、`mla_preprocess`、`batch_matmul_transpose` | `csrc/moe|attention|gmm|mc2/<op>/{op_host,op_kernel}` |
| 编译 | setup.py → CMake `ascendc_library(vllm_ascend_kernels)`，随 `vllm_ascend_C` 一起编（`CMakeLists.txt:55-78`） | `csrc/build.sh --pkg --ops=<op1,op2> --soc=ascend910b` → `.run` 包 → 安装到 `vllm_ascend/_cann_ops_custom/vendors/`（`build_aclnn.sh`） |
| 注册 | `torch_binding.cpp:2833` `TORCH_LIBRARY_EXPAND(CONCAT(_C,_ascend), …)`；aclgraph 需在 `torch_binding_meta.cpp` 写 meta 实现 | 同左（wrapper 在 torch_binding.cpp，aclnn 实体在 vendors 包） |
| Python 入口 | `vllm_ascend/lora/lora_ops.py`（bgmv/sgmv）、`vllm_ascend/ops/mla.py` 等 | `vllm_ascend/ops/**`，需 `enable_custom_op()`（`vllm_ascend/utils.py:521`） |
| 增量重编 | 只能整体重编扩展（setup.py 备注：ascendc kernel 暂不支持 ccache/ninja 增量） | **支持单算子增量**：`build.sh --ops=rms_norm_cast` 只编一个算子 |
| 910B 支持 | ✅（ascend310p/950 会跳过 `vllm_ascend_kernels`） | ✅ 44 项（见 §3 清单） |

> 注：`*_metadata` 子目录是 host-only 配套算子（infer-shape/tiling 用），没有 device kernel。

---

## 3. 910B（ascend910b）生效的自定义算子清单

`csrc/build_aclnn.sh` ascend910b 分支共 44 项（含 metadata 配对）：

- **KV/Attention 写入与变换**：`transpose_kv_cache_by_block`、`store_kv_block(+metadata)`、`scatter_nd_update_sk`、`copy_and_expand_eagle_inputs`、`inplace_partial_rotary_mul`
- **归一化/激活量化**：`rms_norm_cast`、`add_rms_norm_bias`、`rms_norm_dynamic_quant`、`dequant_swiglu_quant`、`dequant_situ_quant`
- **MoE**：`moe_gating_top_k(+_hash)`、`hc_pre`、`hc_post`、`grouped_matmul_swiglu_quant(_v2/_weight_nz_tensor_list)`（gmm/）
- **稀疏注意力/索引器（DSA、MLA 系列）**：`lightning_indexer(_quant)`、`vllm_quant_lightning_indexer(+metadata)`、`quant_lightning_indexer_v2(+metadata)`、`sparse_flash_attention`、`sparse_flash_mla(+metadata)`、`sparse_attn_sharedkv(+metadata)`、`kv_quant_sparse_flash_attention`、`sparse_attention_score`、`k2q_csr`、`msa_index_score`、`fused_sparse_attention_overlap`、`compressor(+metadata)`
- **线性注意力/GDN/KDA**：`recurrent_gated_delta_rule`、`recurrent_kda`、`chunk_fwd_o`、`chunk_gated_delta_rule_fwd_h`、`chunk_kda_fwd`、`kda_gate_cumsum`、`kda_layout_swap12`
- **其他**：`causal_conv1d`

路径 ① 的 6 个 AscendC kernel：`bgmv_expand`、`bgmv_shrink`、`sgmv_expand`、`sgmv_shrink`、`mla_preprocess`、`batch_matmul_transpose`。

---

## 4. 本机开发-测试链路（静态结论）

三层测试设施，全部可在本机 910B3 上运行：

1. **NPU 单卡算子测试**（与算子几乎一一对应，最适合做优化回归）：
   `tests/e2e/nightly/single_node/ops/singlecard_ops/`，共 46 个文件。例如
   `test_bgmv_expand.py`（含 CPU 参考实现 `bgmv_expand_cpu_impl` 做数值对照）、
   `test_rms_norm_cast.py`（对照 `torch_npu.npu_rms_norm`，另有 NPUGraph 捕获测试）、
   `test_transpose_kv_cache_by_block.py`、`test_dequant_swiglu_quant.py` 等。
2. **csrc 主机侧 UT**（不需要 NPU，可跑 ophost/tiling 逻辑 + ASAN/覆盖）：
   `csrc/build.sh -u [--ophost_test|--opapi_test|--opgraph_test] [--noexec] [--cov]`，
   也支持 `--ops=` 过滤。
3. **tests/ut**（CPU mock，验证 Python 侧封装/分发逻辑）。

**优化迭代闭环（算子级）**：

```bash
# 1. 改 op_kernel/*.cpp 后，只编目标算子并安装到 repo 内目录
cd csrc && bash build.sh --pkg --ops=<op_name> --soc=ascend910b
./build/cann-ops-transformer*.run --install-path=$PWD/../vllm_ascend/_cann_ops_custom
# 2. 直接跑对应单卡测试
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_<op_name>.py
```

新算子接入流程（若新增）：见 `docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md`
（建目录 → build_aclnn.sh 加 SOC 列表 → torch_binding.cpp 绑定 → torch_binding_meta.cpp 写 meta）。

---

## 5. 算子复杂度分级

| 级别 | 算子（kernel 代码量级） | 特点 |
|---|---|---|
| 小型（vector/mem-bound，核心逻辑 <300 行） | `rms_norm_cast`(29+头文件)、`add_rms_norm_bias`、`transpose_kv_cache_by_block`(35+头)、`scatter_nd_update_sk`(75+arch22 头)、`inplace_partial_rotary_mul`(322)、`hc_pre/hc_post`、`msa_index_score`、`compressor`、`copy_and_expand_eagle_inputs`(388) | 结构清晰、易改动易验证；多为每层/每步都跑的高频小算子 |
| 中型 | `dequant_swiglu_quant`（apt 1411 行）、`causal_conv1d`、`moe_gating_top_k(_hash)`、`kda_gate_cumsum`(330)、`kda_layout_swap12`、`store_kv_block`、`k2q_csr` | 有分支/多 dtype 模板，中等风险 |
| 大型（matmul/attention 重度） | `sparse_flash_attention`、`sparse_flash_mla`、`lightning_indexer*`、`quant_lightning_indexer_v2`、`kv_quant_sparse_flash_attention`、`mla_prolog_v3_k3`、`grouped_matmul_swiglu_quant*`（依赖 catlass） | 收益最大但风险/工作量最高，往往涉及 tile 策略与 UB 预算 |
| 路径① 内置 | `bgmv_expand`(368)、`bgmv_shrink`(252)、`sgmv_expand`(388)、`sgmv_shrink`(275)、`mla_preprocess`(507)、`batch_matmul_transpose`(824) | LoRA 四件套小而独立；后两个偏大 |

近期 git 活动（2026-06 之后）：`sparse_attention_score`/`msa_index_score`/`quant_lightning_indexer_v2`/`k2q_csr` 变更最密集；`scatter_nd_update_sk` 已有两轮明确 perf 提交（`e2f7e3110` 高性能路径默认化、`4b90a7c49` 用满向量核）——说明这些是团队当前的性能热点。

---

## 6. 重点候选的静态代码观察（优化线索）

1. **`bgmv_expand` / `bgmv_shrink`（LoRA，路径①）**
   - `CopyInX` 里用**逐元素 `GetValue()/SetValue()` 标量循环**把 x 向量复制/广播到 64 float（`csrc/kernels/bgmv_expand.cpp:193-217`），rank=64 时每 token 约 64+ 次标量访存，向量核上这是典型可替换点（`Brcb`/向量广播、或一次性 DataCopy 对齐拷贝）。
   - W tile 8192 元素、double buffer 已具备；Compute 走 BlockReduceSum+PairReduceSum 分 rank 展开。
   - 自带 CPU 参考实现 + 单卡测试，**改动→验证闭环最短**。
   - `bgmv_shrink`/`sgmv_*` 同族结构，优化经验可平移。

2. **`rms_norm_cast`（MoE 路由前的高频点）**
   - kernel 仅 29 行 + 头文件实现，输出 bf16 + fp32 双份（供路由用）；hidden=7168 每层多次调用，decode 小 batch 场景是典型 bandwidth-bound。
   - 有现成对照（`torch_npu.npu_rms_norm`）和 NPUGraph 测试；A5 上曾融合/回退（`033198cd6`/`b0c49dc7a`），说明被持续关注。

3. **`transpose_kv_cache_by_block`（P/D、量化 KV 布局相关）**
   - 35 行入口 + `full_load.h`/`general.h` 双路径，mem-bound 搬运类算子；对齐/分块策略是常规优化抓手（vector 搬运宽度、NZ 布局）。

4. **`scatter_nd_update_sk`（KV offload 写回）**
   - 已优化两轮：HP 路径每核独立完成 LinearIndex+Scatter 消除 SyncAll、tiling 用满向量核。
   - 残余点：`tilingKey 30`（int64 大索引）仍走旧 `LargeIndexKernel` 回退路径，是明确的静态可改进项（int64→int32 线性索引不可表达时的性能悬崖）。

5. **`inplace_partial_rotary_mul` / `copy_and_expand_eagle_inputs` / `hc_pre/hc_post`**
   - 小型搬运/逐元素类，单测齐备；近期也有 perf 类提交（`1ffadbc7a` 把 negate_sin 下沉进 kernel），说明这类"小算子链路"在持续榨性能。

6. **`dequant_swiglu_quant`（MoE 共享专家热点，apt 1411 行）**
   - 中型里收益最明确的一个：反量化+SwiGLU+量化三合一，MoE 前向必经；已有多 dtype/多模式分支，优化空间在 tile 与两阶段 cast 的融合策略，但改动面大。

7. **大型算子（`sparse_flash_*`、`lightning_indexer*`、`grouped_matmul_swiglu_quant*`）**
   - 收益上限最高（prefill/decode 主耗时），但涉及 catlass/tile 预算，且近期迭代密集，适合有明确 profile 证据后再切入。

---

## 7. 优化选题建议（按"收益/风险/验证成本"排序）

| 优先级 | 选题 | 理由 |
|---|---|---|
| ★★★ | `bgmv_expand`(+shrink) 标量广播优化 | 自包含、测试闭环最短、有明确静态瓶颈线索；分支名为 910B_perf 的理想切入点 |
| ★★★ | `rms_norm_cast` / `add_rms_norm_bias` 带宽优化 | decode 高频、结构小、有 torch_npu 基线可直接对拍 |
| ★★☆ | `scatter_nd_update_sk` int64 大索引路径 | 已有成熟 HP 路径可借鉴，回退路径是明确短板 |
| ★★☆ | `transpose_kv_cache_by_block` 搬运优化 | mem-bound、逻辑简单，适合练手/快速出成果 |
| ★☆☆ | `dequant_swiglu_quant` tile 融合 | MoE 主路收益大，但改动面与回归成本高 |
| ★☆☆ | 大型 attention/indexer 算子 | 先用 msprof/torch_npu profiler 拿到占比证据再定 |

**方法论提醒**：以上"优化空间"均为静态推断，动手前应先用 `msprof` / `torch.npu.profiler`（或 ACLProf）确认该算子在目标场景（模型/batch/序列）中的实际占比与瓶颈类型（vector pipe / MTE / GM 带宽 / 同步等待），避免优化到非热点。

---

## 8. 命令速查

```bash
# 单算子重编+安装（aclnn 路径②）
cd csrc
bash build.sh --pkg --ops=rms_norm_cast --soc=ascend910b -j$(nproc)
./build/cann-ops-transformer*.run --install-path=$PWD/../vllm_ascend/_cann_ops_custom

# 跑对应算子的单卡 NPU 测试
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_rms_norm_cast.py

# 主机侧 UT（ophost/tiling 逻辑，无需 NPU）
bash build.sh --ophost_test --ops=rms_norm_cast --noexec

# 路径①（bgmv/sgmv 等）改完需重编扩展
pip install -e . --no-build-isolation   # 或按 setup.py 的 CMake 流程
```

> 本文件为静态分析产物（未运行任何 NPU 任务），随编译完成可用 §8 的测试命令逐一验证。
