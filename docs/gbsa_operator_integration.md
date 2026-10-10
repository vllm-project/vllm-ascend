# GBSA 算子接入 vLLM-Ascend 过程文档

Status: 源码、Ascend 950 自定义算子包和 Python 调用链已完成构建验证；NPU 数值与性能验证待完成

## 1. 目标与范围

本文记录将 `ops-transformer_gbsa/attention/generic_block_sparse_attention`
接入当前 vLLM-Ascend 工作树的设计、实现、环境、验证过程和注意事项。

本次范围包括：

1. 将 GBSA 的 op host、tiling、op kernel 及依赖源码放入
   `csrc/attention/generic_block_sparse_attention`。
2. 将算子加入 `csrc/build_aclnn.sh` 的对应 SoC 构建列表。
3. 复用 vLLM-Ascend 的 `_C_ascend` 动态库，把 ACLNN 接口注册为
   `torch.ops._C_ascend.*`。
4. 提供 Python 包装接口和 Meta 实现，使 eager、FakeTensor 与图捕获阶段能看到
   正确的 schema、输出 shape 和 dtype。

本次不包含把 GBSA 自动接进某个具体模型 attention backend 的调度逻辑。模型层若要
实际选用 GBSA，还需根据模型的稀疏索引生成方式，把 `sparse_block_idx`、
`sparse_block_count`、paged KV cache 和序列元数据传给本文提供的 Python 接口。

## 2. 固定版本与环境

| 组件 | 版本 |
| --- | --- |
| 工作目录 | `/mnt/share/j00976700/code_qwen38_next_gbsa` |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | detached `e04af1d22404a7afff2a1f70d3a1add8e038358b` |
| GBSA 源仓 | `feat/lch/gbsa`，`5630eaf4afed34eb0740d2ac2d9f053a6f4dd453` |
| 构建 SoC | `ascend950` / `Ascend950PR_9599` |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| 构建容器 | `vllm_ascend_jt_qwen38next_a5_new` |
| 容器镜像 | `vllm-ascend:qwen38next-a5-new-base` |
| 续编验证 CANN | `/usr/local/Ascend/cann-9.2.0` |
| 续编验证镜像 | `vllm-ascend:dev-26.2.0.day20261008-A5-py311-Ubuntu24.04-lts-aarch64` |
| 主机 Python | 3.13.9；未安装 `torch`/`torch_npu`，不能用于 `_C_ascend` 编译验证 |
| 容器构建栈 | Python 3.11.10、PyTorch 2.10.0、torch_npu 2.10.0.post4、CMake 4.4.3、GCC 12.3.1 |

版本必须固定。GBSA 的 host schema、tiling key、kernel 模板和 PyTorch 包装是一套协议，
不能只复制某个 `.cpp` 后再混用其他提交的文档或 Python 接口。

### 2.1 容器与目录映射

构建在正在运行的 `vllm_ascend_jt_qwen38next_a5_new` 容器中完成。该容器把主机的
`/mnt`、`/home`、`/data` 和 `/root/.cache` 原路径映射进容器，因此本文的
`/mnt/share/j00976700/code_qwen38_next_gbsa` 在容器内仍是同一路径，代码修改和构建产物
会直接落到共享目录，而不是只存在于容器可写层。

容器还映射了 `/usr/local/Ascend/driver`、firmware、`npu-smi`、DCMI 和
`/etc/ascend_install.info`，用于访问主机驱动、固件、设备管理和 SoC 信息。CANN
toolkit 本体来自容器镜像中的 `/usr/local/Ascend/cann-9.1.0`；它不是从工作树生成的，
也不应提交到仓库。

## 3. GBSA 在做什么

GBSA（Generic Block Sparse Attention）是基于 CATLASS 模板库实现的块稀疏注意力。
它不让每个 query token 与全部历史 KV 做 dense attention，而是沿序列轴把 Q、K、V
组织为块，并只计算每个 Q 块选中的 KV 块：

```text
attention_out = softmax(scale * Q * K_selected^T + mask) * V_selected
```

- `sparse_block_idx`：每个 Q 块选择哪些 KV 块。
- `sparse_block_count`：每个 Q 块实际有效的选择数量；索引尾部可作为 padding。
- `block_table`：Paged KV cache 的逻辑页到物理页映射。
- `metadata`：由 metadata 算子根据本次输入、布局、稀疏 pattern 和属性生成的 1024
  元素 `int32` 设备 tensor，主算子必须配套使用。

调用是严格的两阶段流程：

```text
sparse pattern + sequence metadata + attributes
                  |
                  v
GenericBlockSparseAttentionMetadata -> metadata
                  |
                  v
GenericBlockSparseAttention(Q, K, V, ..., metadata) -> output
```

## 4. 特殊点与优势

### 4.1 主要优势

1. **计算量随实际选择的 KV 块增长。** 当 `topK` 远小于总 KV 块数时，可避免大量
   无效的 QK 和 PV 计算。
2. **减少 KV 读取流量。** 未选中的 KV block 不进入主计算路径，长上下文场景收益尤其
   明显。
3. **块粒度适合 NPU。** 相比任意 token 稀疏，固定块更容易形成连续 DMA 和矩阵计算，
   能利用 Cube/Vector 流水。
4. **原生适配 paged KV cache。** `PA_BBND` 路径通过 `block_table` 访问物理页，不需要
   先把稀疏 KV 拼成大而连续的 dense tensor。
5. **支持 GQA。** Q head 数可以大于 KV head 数，当前要求 `Nq % Nkv == 0`，group size
   不超过 128。

### 4.2 它不是无条件更快

收益取决于稀疏率、序列长度、block size、索引生成成本以及访存局部性。短序列、接近
dense 的稀疏 pattern 或高度离散的块访问，可能无法覆盖 metadata/tiling 与调度开销。
因此只能在正确性通过后，用相同输入和相同稀疏 pattern 做基线对比，不能仅凭“稀疏”
宣称性能提升。

### 4.3 当前提交的关键约束

- 正式支持文档当前以 `layout_q="TND"`、`layout_kv="PA_BBND"` 为主。
- `head_dim` 当前为 128；paged KV page size 为 128，且应等于 `block_shape[1]`。
- 通常使用 `block_shape=[1, 128]`；Arch35 非量化 FP16/BF16 路径允许第二维 1～256。
- 当前 `topK <= 256`，且必须大于等于 `sparse_block_count` 的最大值。
- 当前 packed GQA pattern 要求 group 内 Q heads 共享稀疏 pattern。
- 当前正式约束以 causal mask (`mask_mode=1`) 为主。
- A5 的 TND 非 paged 路径在源提交中标注为“尚未运行构建或设备验证”，不能当作已支持
  能力对外承诺。

## 5. 为什么需要绑定，但不新增独立 pybind11 模块

结论：**需要 Python/C++ 绑定；不应照搬上游的第二个 `PYBIND11_MODULE`。**

上游 `torch_extension` 使用 `cann_ops_transformer.OpBuilder` 编译独立 pybind11 模块。
vLLM-Ascend 已有统一动态库 `_C_ascend`，其惯用接入方式是：

1. C++ adapter 使用 `EXEC_NPU_CMD` 调 ACLNN。
2. `csrc/torch_binding.cpp` 用 `TORCH_LIBRARY` 定义 schema，并注册 PrivateUse1 实现。
3. `csrc/torch_binding_meta.cpp` 注册 Meta 实现。
4. Python 使用 `torch.ops._C_ascend.<op>` 调用。

采用统一链路的原因：

- 避免第二份 `.so`、第二套加载时机和额外的 `cann_ops_transformer` 运行时依赖。
- vLLM-Ascend 导入 `_C_ascend` 后 schema 一次性可见。
- Meta 实现能服务 `torch.compile`、FakeTensor 和图捕获的 shape 推导。
- 打包、ABI、错误处理和其他 vLLM-Ascend ACLNN 自定义算子保持一致。

### 5.1 相关概念及本次是否使用

| 概念 | 含义 | 本次接入中的作用 |
| --- | --- | --- |
| `csrc` | Python 工程中保存 C/C++、Ascend C 和构建脚本的源码目录 | 使用。GBSA 源码、ACLNN adapter、Torch 注册和自定义算子构建入口都在这里 |
| CMake | 生成本机编译规则并组织目标、依赖、编译参数和安装步骤的构建系统 | 使用。既构建 CANN custom-op，也构建 `vllm_ascend_C` 动态库 |
| ACLNN | CANN 面向算子的 C 接口；通常包含 workspace 查询和执行两阶段 API | 使用。adapter 最终调用 `aclnnGenericBlockSparseAttention*` |
| pybind11 | 把 C++ 函数/类型暴露给 Python 的绑定库 | 间接使用。vLLM-Ascend 统一扩展依赖 pybind11，但 GBSA 没有新增独立 `PYBIND11_MODULE` |
| `TORCH_LIBRARY` | PyTorch dispatcher 的 schema/实现注册机制 | 使用。把 GBSA 注册成 `torch.ops._C_ascend.*`，这是本次实际的算子入口 |
| CATLASS | 面向 Ascend AI Core 的模板化高性能算子基础库，作用类似矩阵/搬运/流水模板层 | 使用。GBSA kernel 依赖其模板；仓库通过 `csrc/third_party/catlass` 子模块固定版本 |
| TLA | GBSA 随源码带入的 layout、tensor、tuple 和编译期数值辅助层 | 使用。它位于算子 `op_kernel/tla`，不是 Python 依赖包 |
| op host | 在 CPU 侧完成算子定义、shape 推导、参数校验和 tiling 的代码 | 使用。生成 ACLNN/op proto，并为 device kernel 准备执行参数 |
| op kernel | 在 NPU AI Core 上实际执行计算的 Ascend C/C++ 代码 | 使用。编译后形成 SoC 专用 `.o` 和 kernel `.json` |
| tiling | 根据 shape、dtype、SoC 资源把一次算子拆为可执行 tile，并选择 tiling key | 使用。host 侧生成 tiling 数据，kernel 按相同协议解释 |
| metadata | 本算子第一阶段生成的固定 1024 个 `int32` 设备数据，编码本次执行所需的派生调度信息 | 使用。它不是 Python package metadata，也不是可跨输入复用的静态配置 |
| Meta kernel | 只推导输出 shape/dtype/device、不执行 NPU 数值计算的 PyTorch 实现 | 使用。服务 FakeTensor、`torch.compile` 和图捕获 |

需要区分三个容易混淆的“绑定”：pybind11 负责加载统一 C++ 扩展，`TORCH_LIBRARY`
负责把算子注册给 PyTorch dispatcher，ACLNN adapter 负责从 PyTorch tensor 进入 CANN
算子。GBSA 复用前两者的现有框架，只新增第三层 adapter 和对应 schema/Meta 注册。

## 6. 代码改动

### 6.1 核心算子目录

把源目录中的构建核心复制到：

```text
csrc/attention/generic_block_sparse_attention/
```

保留 `CMakeLists.txt`、`op_host`、`op_kernel`、顶层 README 及 kernel 依赖目录。未复制
上游 `docs`、`examples`、`tests` 和 `torch_extension`，因为它们不是 vLLM-Ascend
自定义算子包的编译输入，且 `torch_extension` 会引入重复的扩展加载方案。

不能只复制顶层 kernel `.cpp`。该实现通过相对路径依赖 `op_kernel/attn_infra`、
`op_kernel/tla` 和架构专用实现，少一个子目录通常要到 device kernel 编译阶段才报错。

### 6.2 CANN 9.1 host 兼容

源文件 `op_host/generic_block_sparse_attention_tiling.cpp` 引用了当前 CANN 环境不存在的
`op_host/tiling_base.h`。接入版本只补了该头提供的 `ASCENDC_EXTERN_C` 宏兼容定义，未改
tiling 算法。该改动必须由 host/tiling 编译验证，不能用删除 include 后“看起来能编译”
作为结论。

### 6.3 自定义算子构建列表

在 `csrc/build_aclnn.sh` 的 `ascend910b`、`ascend910_93` 和 `ascend950` 列表加入：

```text
generic_block_sparse_attention
```

目录解析函数已支持 `csrc/attention/<op_name>`，无需把 GBSA 错放到 `csrc/moe`。这里
“模仿 causal_conv1d”指的是沿用相同的 CANN 自定义算子目录与 CMake 组织方式，而不是
要求 attention 算子归类到 moe。

### 6.4 C++ adapter

新增：

```text
csrc/attention/generic_block_sparse_attention/generic_block_sparse_attention_torch_adpt.h
```

adapter 提供两个函数：

- `npu_generic_block_sparse_attention_metadata`
- `npu_generic_block_sparse_attention`

它负责分配输出、规范 string/IntArrayRef 参数，并调用：

- `aclnnGenericBlockSparseAttentionMetadata`
- `aclnnGenericBlockSparseAttention`

量化模式下若未指定 `attention_out_dtype` 会直接报错；非量化模式默认沿用 `q.dtype`。

### 6.5 Torch schema 与 Meta

`csrc/torch_binding.cpp` 注册 `_C_ascend` schema 和 PrivateUse1 实现；
`csrc/torch_binding_meta.cpp` 注册 Meta 实现。

Meta 输出约定：

- metadata：`[1024]`，`int32`，Meta device。
- attention output：shape 与 Q 相同，dtype 按 `attention_out_dtype` 或非量化 Q dtype。
- `softmax_lse`：关闭时为 `[0]`；打开时为 FP32，并按 Q layout 推导 shape。

### 6.6 Python API

新增 `vllm_ascend/ops/gbsa.py`，提供枚举和两个 typed wrapper。典型调用：

```python
from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)

metadata = generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    [query_block_size, kv_block_size],
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
)

output, softmax_lse = generic_block_sparse_attention(
    q,
    k,
    v,
    sparse_block_idx,
    sparse_block_count,
    [query_block_size, kv_block_size],
    metadata=metadata,
    cu_seqlens_q=cu_seqlens_q,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    block_table=block_table,
    softmax_scale=softmax_scale,
)
```

## 7. 最容易出错的地方

1. **metadata 不能缓存跨调用复用。** shape 一样不代表布局、mask、量化、窗口、序列长度
   和稀疏 pattern 一样；metadata 与主算子参数不一致可能产生错误寻址。
2. **`cu_seqlens_*` 与 `seqused_*` 语义不同。** 前者描述存储边界，后者描述实际有效
   长度；不能用有效长度重新解释 sparse block 编号。
3. **Paged KV 参数必须成套。** `PA_BBND` 需要正确的 `block_table` 和 `seqused_kv`，
   不应传 TND 路径使用的 `cu_seqlens_kv`。
4. **索引 dtype/设备/连续性。** sparse index/count、block table、metadata 以及可选量化
   scale 必须满足 op host 定义，不能依赖隐式 CPU 到 NPU 拷贝。
5. **输出 dtype。** 量化路径必须明确传 `attention_out_dtype`；否则 adapter 主动拒绝。
6. **不要复制独立 pybind11 扩展。** 否则会出现重复 schema、动态库加载顺序和 ABI 风险。
7. **不要遗漏 Meta。** eager 能运行不等于图模式可用；缺 Meta 会在 FakeTensor/compile
   阶段失败。
8. **构建环境要完整。** `opc`、TVM 动态库、CANN Python 包和 compiler 的 PATH/
   `LD_LIBRARY_PATH` 必须同时正确。
9. **关注主机时间漂移。** 本次主机出现约数分钟 clock skew，Make 会警告增量结果可能
   不完整；正式发布前应在时间同步后做一次干净构建。

## 8. 可复用接入流程

1. 固定源算子、vLLM、vLLM-Ascend 和 CANN 版本。
2. 审计 op definition、ACLNN 参数顺序、tiling、kernel include graph、dtype/layout 与
   metadata 契约。
3. 将完整核心目录复制到语义正确的 `csrc/<category>/<op>`。
4. 对照同类算子接入 CMake 和 `build_aclnn.sh`，保留原算子的 ACLNN target 类型。
5. 把上游 extension 翻译为 `EXEC_NPU_CMD` adapter，不新增第二个 pybind11 模块。
6. 在 `_C_ascend` 中注册 schema、PrivateUse1 和 Meta。
7. 增加薄 Python wrapper，清楚暴露必需的两阶段调用。
8. 先构建单个 custom op，再构建完整 custom-op 包，最后编译 editable `_C_ascend`。
9. 检查 schema 存在性和 Meta shape/dtype。
10. 在 NPU 上用 dense PyTorch reference 验证最小 FP16/BF16、GQA、paged KV、边界
    topK 和图模式用例。
11. 正确性全部通过后再做性能测试，并报告稀疏率、序列长度、block shape 和基线。

## 9. 本次验证记录

已完成：

- `git diff --check` 通过。
- 源目录与目标目录逐文件核对；差异仅为刻意排除的非构建目录、新增 adapter，以及
  CANN 9.1 host 兼容改动。
- CMake 成功发现 GBSA 单算子目标。
- op definition、op host、opapi、tiling 与 `_rt2.0` host library 已编译通过。
- 四个 Ascend 950 GBSA kernel 变体均已通过 `opc`/`ccec` 编译，生成对应 `.o` 与
  `.json`。
- 单算子 `.run` 包已成功生成并通过 SHA256 自校验。
- 为避免覆盖仓内全量 `_cann_ops_custom`，单算子包安装到隔离目录
  `csrc/build/gbsa_install`；host、opapi、op proto 与 kernel 文件均安装成功。
- 在 PyTorch 2.10.0、torch_npu 2.10.0.post4、Python 3.11.10 容器中，使用精确 SoC
  `ascend950pr_9599` 成功编译并链接含 GBSA 的 `vllm_ascend_C`。
- `_C_ascend` 两个 schema 可见；Meta smoke 覆盖 metadata shape/dtype、BF16 TND
  output、启用/关闭 LSE，以及 FP8 输入指定 BF16 输出，全部通过。
- 2026-10-08 在同一工作树、同一容器名但 CANN 9.2 镜像中重新验证：只构建 GBSA 时
  四个 Ascend 950 kernel、单算子 `.run`、隔离安装、`vllm_ascend_C` 链接和
  schema/Meta smoke 均通过。
- CANN 9.2 产物保存在仓库外的
  `.gbsa-build-artifacts/gbsa-cann92-20261008-130740`；单算子包 SHA256 为
  `73065844e6ceb3d4e1b3d85f97d01bbb4799af005a6861b081b2e0e3dd0fe910`，
  `_C_ascend` SHA256 为
  `0504aca7a7c73a11b7dd19ae9182539d214848b61fd27e045d2ab16727af8f2e`。

构建过程中定位并处理：

- 旧 CMake cache 固定了不存在的 Ninja 路径；通过保留旧 build 目录并干净重配解决。
- `opc` 位于 `aarch64-linux/bin`，默认 CANN PATH 未包含该目录。
- TVM 库位于 `aarch64-linux/lib64`，需要加入 `LD_LIBRARY_PATH`。
- 被超时终止的 `opc` 留下 stale `kernel_meta.lock`；确认 PID 不存在后将锁文件改名保留，
  再做增量构建。
- `EXEC_NPU_CMD` 的类型转换接收可绑定的左值引用，不能直接传
  `const_cast<char *>(layout.c_str())` 临时表达式；必须先保存为局部 `char *`。
- `_C_ascend` 的 CMake 配置不能使用泛化的 `ascend950`，当前 CANN 要求精确 SoC
  `ascend950pr_9599`。
- 仅 GBSA 的 `.run` 包不能安装到正式 `_cann_ops_custom` 后覆盖其他算子；正式发布时要
  由全量 `build_aclnn.sh` 生成组合包。
- CANN 9.2 全量 editable 构建曾在约 `394/616` 失败，首个失败目标为既有
  `causal_conv1d`，同时 `chunk_gated_delta_rule_fwd_h` 的旧 op-info 触发 CANN `opc`
  的 `TypeError: 'NoneType' object is not subscriptable`。日志中 GBSA host/proto 已通过，
  因而该失败不能归因于 GBSA；续编时隔离失败的全量 build，只选择
  `generic_block_sparse_attention`，随后目标链路全部通过。

仍需完成并记录结果：

- NPU 最小正确性、paged KV、GQA、graph capture/replay 和代表性边界用例。
- 与 dense reference 的数值误差，以及与基线 attention 的性能对比。

当前原生会话未提供 VAWS `context_file`，因此没有绕过受管资源所有权直接占用现有 NPU
容器执行设备测试。拿到有效任务上下文后，应按第 8 节第 10～11 步继续。

在上述设备验证完成前，本接入可表述为“代码和构建链已接入”，不能表述为“已完成
生产验证”或“已有确定性能收益”。

## 10. 8000+ 新文件、提交边界与源码安装

### 10.1 为什么 `csrc` 下出现 8000+ 新文件

现场统计如下：

| 路径 | 文件数 | 大小 | Git 状态 |
| --- | ---: | ---: | --- |
| `csrc/build.pre_gbsa_integration` | 8731 | 611 MiB | 未跟踪，因目录名不匹配现有忽略规则 |
| `csrc/build` | 3492 | 221 MiB | 被 `.gitignore` 的 `build/` 规则忽略 |
| `csrc/attention/generic_block_sparse_attention` | 91 | 约 3 MiB | 本次应审阅、应提交的 GBSA 源码 |

`build.pre_gbsa_integration` 是开始 GBSA 接入前，把当时的 `csrc/build` 重命名保留下来的
旧构建快照。其 `CMakeCache.txt` 的 `ASCEND_OP_NAME` 列表没有 GBSA，安装前缀甚至仍指向
旧的 `code_qwen38_next_gsqa` 工作目录；因此它不是 GBSA 编译所需依赖，更不是应随代码
提交的内容。

原目录名 `build` 会被仓库 `.gitignore` 忽略；改名为 `build.pre_gbsa_integration` 后不再
匹配 `build/`，所以 Git 将快照中的约 8509 个文件连同其他真正源码一起显示为 8000+
新增文件。里面主要是 CMake/Ninja 状态、依赖文件、自动生成源码、host `.o`、Ascend
kernel `.o`/`.json`、打包临时目录和安装包内容。

提交 PR 时只应加入源码、构建描述、绑定、Python wrapper、测试和文档。不要 `git add`
以下生成物：

```text
csrc/build.pre_gbsa_integration/
csrc/build/
csrc/build_out/
csrc/output/
vllm_ascend/_cann_ops_custom/   # 除仓库原有 .gitkeep 外
```

`build.pre_gbsa_integration` 可留在仓库之外作短期诊断备份，或确认不再需要后删除；无论
选择哪一种，都不影响别人从干净 clone 编译 GBSA。

### 10.2 干净 clone 能否直接按 vLLM、vLLM-Ascend 顺序安装

**从构建链看可以，但“直接安装”以完整构建环境和提交完整源码为前提，不依赖上述
8731 个旧构建文件。** 当前 `setup.py` 在 `COMPILE_CUSTOM_KERNELS=1`（默认值）时会先调用
`csrc/build_aclnn.sh` 构建并安装整组 ACLNN custom-op，再通过 CMake 构建统一的
`vllm_ascend_C`；GBSA 已进入对应 SoC 的 custom-op 列表，因此会被自动带上。

建议在与目标 SoC/CANN/PyTorch 匹配的容器中使用以下流程：

```bash
cd vllm
git submodule update --init --recursive
python -m pip install setuptools-rust
VLLM_TARGET_DEVICE=empty \
  python -m pip install -v --no-build-isolation -e .
cd ..

cd vllm-ascend
git submodule update --init --recursive

# Ascend 950 当前验证环境；其他机器应改成自身的精确 SoC。
export SOC_VERSION=ascend950pr_9599
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export PATH="$ASCEND_HOME_PATH/aarch64-linux/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/aarch64-linux/lib64:$LD_LIBRARY_PATH"

python -m pip install -v --no-build-isolation -e .
```

用户给出的 `pip install setuptools_rust` 也能被 pip 规范化为同一项目名，文档采用其标准
发行名 `setuptools-rust`。`--no-build-isolation` 表示复用当前环境，**不表示跳过依赖
准备**；运行前仍需具备兼容的 `setuptools`、`setuptools-scm`、`wheel`、CMake、Ninja、
pybind11、编译器、CANN、PyTorch 和 torch_npu。Ascend 950 还需要 `opc` 所在的
`aarch64-linux/bin` 和 TVM 等库所在的 `aarch64-linux/lib64` 可见。

提交前还必须确认以下内容确实进入 vLLM-Ascend commit：

1. `csrc/attention/generic_block_sparse_attention` 的 91 个源码/构建描述文件。
2. `csrc/build_aclnn.sh` 中目标 SoC 的 GBSA 条目。
3. `csrc/torch_binding.cpp`、`csrc/torch_binding_meta.cpp` 的 schema、NPU 和 Meta 注册。
4. `vllm_ascend/ops/gbsa.py` 及相应测试/文档。
5. `csrc/third_party/catlass` 子模块指针可获取；不要把本机构建后的 CATLASS 产物提交。

若设置 `COMPILE_CUSTOM_KERNELS=0`，安装会跳过 ACLNN custom-op 和
`vllm_ascend_C` 编译，这只适用于无 NPU 的文档/部分 UT 环境，不能用来验证或运行 GBSA。
安装完成后至少应执行 schema/Meta smoke；正式交付还需完成第 9 节列出的 NPU 数值和
性能验证。本文已经分别验证 custom-op 包和 editable `_C_ascend` 的关键构建步骤，但尚未
在全新 clone/空 build 目录中原样执行上述整段命令，因此在完成一次 clean-clone smoke
前，应将结论表述为“安装链路具备该能力”，而不是“所有环境已验证一键安装成功”。

## 附录 A：算子目录之外的全部代码修改

本附录以远端最终工作树
`/mnt/share/j00976700/code_qwen38_next_gbsa/vllm-ascend` 为准。行号是接入后的文件行号；
若后续继续修改同一文件，行号可能移动，应同时按函数名或 schema 名定位。

### A.1 `csrc/build_aclnn.sh`

| 接入后行号 | 修改内容 | 目的 |
| --- | --- | --- |
| 103 | 在 `ascend910b` 的 `CUSTOM_OPS_ARRAY` 中增加 `generic_block_sparse_attention` | 让 A2 构建流程发现并打包 GBSA |
| 160 | 在 `ascend910_93` 的 `CUSTOM_OPS_ARRAY` 中增加 `generic_block_sparse_attention` | 让 A3 构建流程发现并打包 GBSA |
| 215 | 在 `ascend950` 的 `CUSTOM_OPS_ARRAY` 中增加 `generic_block_sparse_attention` | 让 A5 构建流程发现并打包 GBSA |

只增加了上述三行，没有改动构建脚本的清理、打包或安装逻辑。脚本原有的
`resolve_op_dir` 已包含 `csrc/attention/<op_name>`，因此不需要增加新的目录搜索分支。

### A.2 `csrc/torch_binding.cpp`

| 接入后行号 | 修改内容 | 目的 |
| --- | --- | --- |
| 45 | include `generic_block_sparse_attention_torch_adpt.h` | 让统一 `_C_ascend` 编译单元看到两个 adapter 函数 |
| 3100～3108 | 定义 `npu_generic_block_sparse_attention_metadata` Torch schema | 描述 metadata 算子的 tensor、属性、默认值和返回类型 |
| 3109～3110 | 为 metadata schema 注册 `torch::kPrivateUse1` 实现 | NPU tensor 调度到 C++ ACLNN adapter |
| 3112～3121 | 定义 `npu_generic_block_sparse_attention` Torch schema | 描述主算子的 Q/K/V、稀疏索引、paged KV、量化、mask 和输出 dtype 参数 |
| 3122～3124 | 为主算子注册 `torch::kPrivateUse1` 实现 | NPU tensor 调度到 C++ ACLNN adapter |

这部分没有新增 `PYBIND11_MODULE`。vLLM-Ascend 原有的 pybind11 模块仍只负责加载统一
动态库；GBSA 通过 `TORCH_LIBRARY` 注册到 `torch.ops._C_ascend`。

### A.3 `csrc/torch_binding_meta.cpp`

| 接入后行号 | 修改内容 | 目的 |
| --- | --- | --- |
| 322～339 | 新增 `npu_generic_block_sparse_attention_metadata_meta` | 返回 Meta device 上 `[1024]`、`int32` 的 metadata tensor |
| 341～390 | 新增 `npu_generic_block_sparse_attention_meta` | 推导 attention output 与 softmax LSE 的 shape/dtype |
| 2212～2213 | 注册 metadata 的 Meta 实现 | 支持 FakeTensor、图捕获和 compile 阶段传播 |
| 2214～2215 | 注册主算子的 Meta 实现 | 避免图模式在没有真实 NPU 执行时因缺少 kernel 而失败 |

主算子 Meta 的具体规则如下：

- attention output 的 shape 与 `q` 相同。
- 非量化模式且未指定 `attention_out_dtype` 时，输出 dtype 等于 `q.dtype`。
- 量化模式必须指定 `attention_out_dtype`，否则 Meta 阶段直接报错。
- `return_softmax_lse=False` 时，LSE shape 为 `[0]`、dtype 为 FP32。
- TND 的 LSE shape 为 `[T, N, 1]`；BNSD 为 `[B, N, S, 1]`；其他分支按 BSND
  推导为 `[B, N, S, 1]`。

### A.4 `vllm_ascend/ops/gbsa.py`

这是全新文件，共 142 行，全部属于本次接入。

| 接入后行号 | 修改内容 | 目的 |
| --- | --- | --- |
| 1～5 | License、枚举依赖和 Torch import | Python API 基础依赖 |
| 8～12 | 新增 `GBSAMaskMode` | 提供 `NO_MASK`、`CAUSAL`、`WINDOW` 可读枚举 |
| 14～21 | 新增 `GBSAQuantMode` | 提供非量化、FP8 和 FP4 模式枚举，取值与 ACLNN 一致 |
| 23～72 | 新增 `generic_block_sparse_attention_metadata` | typed wrapper，转调 `torch.ops._C_ascend.npu_generic_block_sparse_attention_metadata` |
| 75～142 | 新增 `generic_block_sparse_attention` | typed wrapper，转调 `torch.ops._C_ascend.npu_generic_block_sparse_attention` |

Python 层没有再次编译或加载上游 `cann_ops_transformer.OpBuilder` 扩展，也没有复制上游
`torch_extension`。wrapper 只负责提供稳定的 Python 签名、枚举转换和统一命名空间调用。

### A.5 明确未修改的外围文件

- `csrc/attention/CMakeLists.txt` 未修改：它本来就会自动遍历包含 `CMakeLists.txt` 的
  attention 子目录。
- `CMakeLists.txt` 和 `setup.py` 未修改：统一 `_C_ascend` 已通过 glob 收集
  `csrc/torch_binding.cpp` 与 `csrc/torch_binding_meta.cpp`。
- `vllm_ascend/ops/__init__.py` 未修改：当前采用按模块显式导入方式，不要求在包级
  `__init__` 重导出 GBSA。
- 未修改任何具体模型 runner 或 attention backend，因此当前交付是“算子与 Python
  调用能力接入”，不是“某个模型自动选择 GBSA”的模型层接入。

## 附录 B：GBSA 源代码版本、落盘位置与目录结构

### B.1 来源版本

| 项目 | 分支/状态 | commit |
| --- | --- | --- |
| GBSA 源仓 `ops-transformer_gbsa` | `feat/lch/gbsa` | `5630eaf4afed34eb0740d2ac2d9f053a6f4dd453` |
| 目标 `vllm-ascend` | detached HEAD | `e04af1d22404a7afff2a1f70d3a1add8e038358b` |
| 配套 `vllm` | 当前工作树版本 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |

源目录：

```text
/mnt/share/j00976700/code_qwen38_next_gbsa/
└── ops-transformer_gbsa/
    └── attention/
        └── generic_block_sparse_attention/
```

目标目录：

```text
/mnt/share/j00976700/code_qwen38_next_gbsa/
└── vllm-ascend/
    └── csrc/
        └── attention/
            └── generic_block_sparse_attention/
```

### B.2 目标目录结构

目标目录当前包含 91 个源码/构建描述文件，结构如下：

```text
generic_block_sparse_attention/
├── CMakeLists.txt
├── README.md
├── generic_block_sparse_attention_torch_adpt.h   # vLLM-Ascend 新增 adapter
├── op_host/
│   ├── CMakeLists.txt
│   ├── generic_block_sparse_attention_def.cpp    # GE/ACLNN op definition
│   ├── generic_block_sparse_attention_infershape.cpp
│   ├── generic_block_sparse_attention_tiling.cpp # host tiling，含 CANN 9.1 兼容改动
│   ├── generic_block_sparse_attention_tiling.h
│   └── op_api/
│       ├── aclnn_generic_block_sparse_attention.cpp
│       ├── aclnn_generic_block_sparse_attention.h
│       ├── generic_block_sparse_attention.cpp
│       └── generic_block_sparse_attention.h
└── op_kernel/
    ├── generic_block_sparse_attention.cpp
    ├── generic_block_sparse_attention_fd_utils.h
    ├── generic_block_sparse_attention_kernel_interface.cpp
    ├── generic_block_sparse_attention_metadata_kernel.h
    ├── generic_block_sparse_attention_tilingkey.h
    ├── kernel_common.hpp
    ├── arch22/                              # A2/A3 架构专用 kernel
    ├── arch35/                              # A5 架构及量化 kernel
    ├── attn_infra/
    │   ├── arch/                            # 架构、资源和跨核同步抽象
    │   ├── detail/                          # 对齐、宏和类型辅助
    │   ├── epilogue/
    │   │   ├── block/                       # online softmax、rescale、mask/index
    │   │   └── tile_common/                 # GM/UB tile copy
    │   ├── gemm/
    │   │   ├── block/                       # QK/PV block MMAD
    │   │   └── tile_common/                 # L1/L0/UB 搬运与 tile MMAD
    │   └── layout/                          # GBSA matrix/vector layout
    └── tla/
        └── numeric/                         # tuple、layout、tensor 和编译期数值工具
```

### B.3 复制范围

从源分支复制了构建 GBSA 所需的顶层 CMake、README、`op_host`、`op_kernel` 及其完整
依赖树。以下源目录没有复制到 vLLM-Ascend：

- `docs/`：上游接口和设计文档，不参与目标工程编译。
- `examples/`：上游独立 ACLNN 示例，不属于 vLLM-Ascend runtime。
- `tests/`：上游测试工程依赖自己的构建布局，后续应转换为 vLLM-Ascend 测试。
- `torch_extension/`：包含独立 `OpBuilder`/pybind11 加载方案，会与 `_C_ascend` 重复。

逐文件 `diff -qr` 结果表明：除附录 C 所列的 adapter 新增和 tiling 兼容修改外，其余
已复制的 GBSA 构建源码与 `feat/lch/gbsa@5630eaf4` 保持一致。

## 附录 C：GBSA 算子目录内部的修改

### C.1 新增 vLLM-Ascend Torch adapter

文件：
`csrc/attention/generic_block_sparse_attention/generic_block_sparse_attention_torch_adpt.h`

这是全新文件，共 130 行，不存在于源分支的核心算子目录中。

| 接入后行号 | 修改内容 |
| --- | --- |
| 20～21 | 定义固定 metadata 大小 1024 和非量化模式常量 |
| 24～54 | 实现 `npu_generic_block_sparse_attention_metadata`：分配 NPU `int32[1024]` 输出，并通过 `EXEC_NPU_CMD` 调用 metadata ACLNN 接口 |
| 39～43 | 将 layout string 保存为有生命周期的 `std::string` 与局部 `char *`；不能把 `const_cast<char *>(c_str())` 临时值直接交给 `EXEC_NPU_CMD` |
| 56～128 | 实现 `npu_generic_block_sparse_attention`：检查必需 tensor 和 block shape、确定输出 dtype、分配 output/LSE、调用主 ACLNN 接口 |
| 78～87 | 对空 Q/K/V、空稀疏索引及非法 `block_shape` 做前置检查 |
| 89～98 | 量化模式强制要求 `attention_out_dtype`；非量化默认使用 Q dtype |
| 100～109 | 根据 TND/BNSD/BSND 分配 LSE，关闭 LSE 时分配 `[0]` FP32 tensor |
| 111～114 | 保存两个 layout `char *` 左值并把 bool LSE 标志转为 ACLNN 所需整数 |
| 115～125 | 严格按上游 ACLNN 参数顺序执行 `aclnnGenericBlockSparseAttention` |

### C.2 `generic_block_sparse_attention_tiling.cpp` 的 CANN 9.1 兼容修改

源分支第 23 行原为：

```cpp
#include "op_host/tiling_base.h"
```

目标环境的 CANN 9.1 不提供该 include 路径，因此目标文件移除该 include，并在接入后
第 24～30 行补充其在本算子中实际需要的宏定义：

```cpp
#ifndef ASCENDC_EXTERN_C
#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif
#endif
```

该修改只处理 `ASCENDC_EXTERN_C` 的可用性，没有修改 GBSA tiling 参数校验、tiling key、
任务划分或 kernel 选择算法。host/tiling library 和 Ascend 950 四个 kernel 变体均已在
当前 CANN 环境编译通过。

### C.3 算子目录内未做的修改

- 没有修改 `generic_block_sparse_attention_def.cpp` 的输入、输出、dtype 或 SoC 声明。
- 没有修改两个 ACLNN op-api 实现及参数顺序。
- 没有修改 Arch22/Arch35 kernel 算法、CATLASS/TLA 基础设施或 online softmax 实现。
- 没有修改 metadata kernel 的格式和固定大小。
- 没有把上游 `torch_extension` 复制进目标目录。

## 附录 D：Windows 本地文档位置

本文件已从远端实现工作树同步到 Windows consumer workspace：

```text
D:/agent-vllm-ascend-workspace/vllm-ascend-workspace/docs/gbsa_operator_integration.md
```

该本地文件是接入过程和改动清单的审阅副本；算子源码实际修改仍位于前述 Linux 远端
`vllm-ascend` 工作树。

## 附录 F：2026-10-08 CANN 9.2 全量 editable 续编

Status: 全量 kernel 构建与 editable 安装通过；设备数值验证未执行

本节以现有远端工作树和容器为准，不代表 clean-clone 或设备数值验证。CANN 为
9.2.0，Python 为 3.11.10，PyTorch 为 2.10.0，torch_npu 为 2.10.0.post7，
精确 SoC 为 `ascend950pr_9599`。未升级、降级或卸载容器依赖。

### F.1 问题计数与修复明细

本轮 CANN 9.2 全量 editable 续编共修复 **5 个源码/构建/安装问题**：
3 个算子构建或 tiling 契约问题、1 个增量构建注册表问题、1 个安装依赖声明问题。
此外修复了 **1 个核验脚本误报问题**，不计入产品问题。
若将核验工具也计入，本轮合计 6 个问题。这里不重复计算前文已完成的 GBSA
接入，也不把两个架构入口、多个依赖声明或多个报错算作不同根因。

旧版 F.1 的六条列表混合了“根因”和“修复动作”：其中原第 1、2 条实际是
同一个 FLA 注册干扰问题，不能分别计数。CANN 的 PATH/LD_LIBRARY_PATH 补齐
属于构建环境配套加固，本轮未单独证明一个独立失败，不另计问题。

| 编号 | 问题 | 修改文件 | 修复方式 |
|---|---|---|---|
| P1 | FLA 的同名算子 schema 干扰本地 OPC 构建 | `csrc/build_aclnn.sh` | 仅在构建子 shell 设置 `FLA_NPU_DISABLE_PTH=1` |
| P2 | chunk kernel 的 tiling 类型未显式提供/注册 | `csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h.cpp` | 显式包含共享 struct，并在两个架构入口注册 |
| P3 | chunk host/kernel tiling 字段不一致 | `csrc/moe/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h` | 删除 host 中未使用且 kernel 不存在的 `hasGk` |
| P4 | 切换算子集合后增量构建仍使用旧注册表 | `csrc/cmake/func.cmake` | 注册表生成改为 always-run target，保留 opbuild 依赖 |
| P5 | 精确依赖声明与容器已有版本冲突 | `requirements.txt`、`pyproject.toml` | 调整允许版本范围，不改容器已安装依赖 |

#### P1：FLA 注册干扰，不是 GBSA kernel 错误

- 根因：FLA wheel 的 `fla_npu_opp_env.pth` 在每次 Python 启动时向
  `ASCEND_CUSTOM_OPP_PATH` 前置自身 vendor；OPC 因而可能读到 FLA 的同名
  schema，而不是当前工作树生成的 schema。chunk 的 FLA schema 需要额外
  logical_* 属性，当前参数不含这些属性，出现
  `TypeError: 'NoneType' object is not subscriptable`。
- 修复：在 `build_aclnn.sh` 的构建子 shell 中使用第三方已有开关
  `export FLA_NPU_DISABLE_PTH=1`。同时在该子 shell 为 PATH 和
  LD_LIBRARY_PATH 补齐 CANN 的 `aarch64-linux/bin`、`aarch64-linux/lib64`。
- 边界：不编辑 FLA wheel 或 .pth，不关闭正常运行时的 FLA 默认注册。
  `causal_conv1d` 未做 kernel 源码修改；其修复落在共同构建环境。
- 证据：隔离后 causal 的 2 个和 chunk 的 4 个 kernel 变体构建通过，
  随后的全量安装也包含这些产物。

#### P2：chunk tiling 类型与注册的 CANN 9.2 编译兼容

- 根因：原代码依赖隐式 tiling 类型/注册，在本次 CANN 9.2 构建中出现
  未注册 tiling struct 或类型未知的错误。
- 修复：在架构头文件之前显式包含
  `chunk_gated_delta_rule_fwd_h_struct.h`；在 310P 和较新架构的两个入口中
  分别添加 `REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleFwdHTilingData);`。
- 配套清理：共享 struct 头文件删除过时的“自动生成”说明，未改变该 struct 的字段。
- 证据：本次 A5 的 4 个 chunk kernel 变体编译通过；入口注册静态回归测试通过。
  310P 入口有静态检查，但本轮没有做 310P 设备构建或运行验证。

#### P3：chunk host/kernel tiling 序列化契约不一致

- 根因：host 宏声明含 `hasGk`，共享 plain struct 不含该字段，且 host
  没有设置它。两侧字段顺序/偏移不能保持一致。这是排查中发现的契约缺陷，
  不宣称本轮已复现由它造成的 NPU 数值错误。
- 修复：删除 host 的 `TILING_DATA_FIELD_DEF(bool, hasGk);`；
  不向 kernel 凭空添加字段，不改变计算公式。
- 证据：新增回归测试逐一比较 host 与 plain struct 的字段类型和顺序；
  测试通过，chunk kernel 编译通过。设备数值正确性仍需单独验证。

#### P4：两算子构建切回全量后注册表未刷新

- 根因：INI 已由 opbuild 更新，但原 CMake
  `add_custom_command(OUTPUT ...)` 规则复用了旧 ops-info JSON。
  两算子版本约 12 KB 的 JSON 缺少全量算子信息，导致
  ChunkKdaFwd/Compressor 解析时落到 CANN 内置的不匹配 schema；
  前者出现输出 13 对 11，后者出现输入 12 对 9 的个数冲突。
  这是注册表生成问题，不另算两个 kernel 源码缺陷。
- 修复：`add_ops_info_target` 改为 `add_custom_target(... ALL)`，
  用 `BYPRODUCTS` 声明 JSON；仍依赖
  `opbuild_gen_default`、`opbuild_gen_inner`、`opbuild_gen_exc`，
  每轮在 opbuild 后重新生成并复制注册表。
- 证据：JSON 更新为全量约 213 KB，完整 347 步构建和安装通过；
  注册表刷新静态回归测试通过。无需删除全部已编译缓存。

#### P5：原始 pip 命令试图更换容器依赖

- 根因：声明要求 `torch-npu==2.10.0.post4`、两个 mem* 包 `==1.2.0`，
  但容器已有 post7、1.3.0。原始 pip 命令会尝试获取/降级依赖，
  不符合“不改依赖版本”的要求。
- 修复：requirements 改为 `torch-npu>=2.10.0.post4,<2.11`、
  `memfabric_hybrid>=1.2.0,<1.4`、
  `memcache_hybrid>=1.2.0,<1.4`；同步 pyproject 的 torch-npu 构建依赖声明。
  没有使用 `--no-deps` 绕过解析，也没有升级、降级或卸载已安装依赖。
- 证据：离线 dry-run 仅计划当前项目；原始 pip 命令最终退出码 0，
  安装前后以相同 pip 口径比较，包版本变更为空。
- 边界：允许版本范围不等于范围内每个版本都经过数值、PD 或 kvpool 验证。

#### V1：核验脚本的重复 distribution 误报（工具问题，单独计数）

安装前快照来自 `pip list --format=json`，初版安装后核验却用
`importlib.metadata.distributions()` 的字典覆盖顺序。在可见路径存在重复
MindStudio distribution 时，两种方式选择了不同版本，误报 26.0.0→26.2.0。
改为安装前后都用同一 Python 的 `pip list --format=json` 后，差异为空。
修复文件为证据目录的 `verify_editable.py`，不是 vllm-ascend 产品源码。

editable 位置不再另计一个编译 bug：原始 `-e .` 成功安装后，项目根目录
会由真实 editable metadata 正常显示；包导入目录另见 F.3。

### F.2 验证证据与边界

- `causal_conv1d` 两个、`chunk_gated_delta_rule_fwd_h` 四个变体均生成 `.o/.json`。
- `tests/ut/ops/test_chunk_fwd_h_tiling_contract.py` 的字段顺序、入口注册和注册表刷新三个测试通过。
- 离线 pip dry-run 通过，仅计划安装当前 `vllm_ascend`，没有依赖变更。
- 全量安装日志与备份位于工作目录外
  `.gbsa-build-artifacts/full-editable-cann92-20261008/`。
- 原始 `pip install -v --no-build-isolation -e .` 全量构建安装退出码为 0，
  日志为 `pip-install-attempt2.log`，退出码为 `pip-install-attempt2.exitcode`。
  未设置 `COMPILE_CUSTOM_KERNELS=0`，未使用 `--no-deps`，没有跳过既有算子。
- `verify-editable.log` 确认 editable direct_url 指向项目根目录，扩展加载通过；
  安装后的 causal/chunk/GBSA 分别包含 2/4/4 个 kernel 对象。
  GBSA metadata Meta 和 BF16、FP8→BF16 的 LSE 开/关四组 Meta 检查通过。
- 安装前后以相同 `pip list --format=json` 口径比较，包版本变更为空。
  初版核验脚本混用 pip 与 importlib 的重复 distribution 选择顺序，曾误报
  CANN 自带 MindStudio 版本差异；改为相同 pip 口径后通过，并非依赖升级。
- `bash -n csrc/build_aclnn.sh`、`git diff --check` 与三个静态回归测试通过。
- NPU 数值、graph、性能验证仍待有效 VAWS 任务上下文。

### F.3 安装位置的含义

`pip list` 的 editable location 应为项目根目录：

```text
/mnt/share/j00976700/code_qwen38_next_gbsa/vllm-ascend
```

Python 包导入位置才是：

```text
/mnt/share/j00976700/code_qwen38_next_gbsa/vllm-ascend/vllm_ascend/__init__.py
```

不应通过移动 setup.py、修改包名或伪造安装元数据，把项目根目录写成包子目录。

本次实测 `pip list` 显示：

```text
vllm_ascend  0.1.dev1+ge04af1d22.d20261008  /mnt/share/j00976700/code_qwen38_next_gbsa/vllm-ascend
```

该版本由当前 Git checkout 与 setuptools-scm 生成，不沿用旧环境的 0.19.1 版本。

## 2026-10-08：补齐自定义 GBSA metadata

Status: dated implementation notes; device execution remains to be verified.

此前自定义 `libcust_opapi.so` 只导出 GBSA 主算子，metadata API 回退到
CANN 内置实现，触发 `headDim currently only supports 128`。这不是 metadata
调用不存在，也不能仅通过修改 head_dim 常量解决。

从 `ops-transformer_gbsa` 的 `feat/lch/gbsa` 分支
`5630eaf4afed34eb0740d2ac2d9f053a6f4dd453` 引入
`csrc/attention/generic_block_sparse_attention_metadata/` 的 host/API、AICPU
kernel、JSON 和 tiling 定义。AICPU CMake 按本仓库三参数
`add_aicpu_cust_kernel_modules` 接口适配。`csrc/build_aclnn.sh` 在 GBSA
所在的 A2/A3/A5 列表中同时加入 metadata。

复用 `csrc/build` 缓存构建完整原算子列表加 metadata，不使用只有两个算子
的安装包覆盖现有算子集，不重编 vLLM 或 Python/C++ 绑定。安装前保留旧
自定义算子目录。安装后应核查自定义库同时导出两个 ACLNN API，并确认
AICPU kernel 和配置已打包；既有服务必须重启才能使用新库。

QSA 适配已显式传入 `softmax_precision=0`，本次不修改该值。新 metadata
源码允许 Arch35 非量化路径的 head_dim 不超过 512，但这不代替真实
head_dim=256 算子正确性及服务请求验证。
