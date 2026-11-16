# K3.1 A3 MLA 真实算子接入说明

更新：2026-11-16。基于 [Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876)，
对接 [mla-release-261116](https://github.com/defeng2008/flash-attention-npu/tree/mla-release-261116)
的 `cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c`。

框架代码已接入真实接口，CPU 回归和指定算子源码的编译通过。用户明确要求不使用占用中的 NPU 做验证，
因此真实 attention 数值、ACL graph、DCP/HCCL、完整 K3.1 及性能结果仍为待验证，不能宣称生产验收完成。

## 接入方式

```text
DSpark / runner 产生设备长度和 Query 分段
  → FlashMLAMetadataBuilder（复用 A5 的缓冲区、刷新和 DCP 流程）
  → A3MLAAdapter（转换 A3 参数和校验 native uint8 metadata）
  → get_scheduler_metadata（AICPU 调度）
  → flash_attn_with_kvcache（MLA 计算）
  → 原有 LSE 合并和 V-up 投影
```

真实算子导入路径仍为 `flash_attn_npu_3.flash_attn_npu_interface`。公开函数名称、参数顺序和四返回值
与原参考接口相同。Q 拆为 RoPE 64 维和内容 512 维；分页 KV 为单 KV head、128-token 算子页；
始终显式传模型的 scale。

## 本轮修改

| 位置 | 修改及原因 |
| --- | --- |
| `attention/a3_mla_sink.py` | external 接受连续的 uint8 metadata，mock 保留私有 int32 记录；不转换 native tiling 字节 |
| 同上 | 提供分离 KV 分配方法，使 A3 DCP current cache 具有内层连续的 512/64 布局 |
| 同上 | native 路径不调用 `torch._assert_async`，因为当前 torch_npu 会将其回退到 CPU；长度值约定由 production builder 构造保证 |
| `attention/flashmla.py` | A3 current 使用 provider 的分离缓存；A5 仍使用原有融合缓存；metadata 仍用 `empty_like` 分配并 `copy_` 刷新 |
| `attention/context_parallel/mla_cp.py` | scatter 分别写两个 component，跨 attention stream 时分别 `record_stream`，防止异步使用期间内存被回收 |
| CPU 测试 | 覆盖 byte ABI、错误缓冲区、native 路径不调用回退断言、A3/A5 DCP 输出及流生命周期 |
| NPU 测试入口 | 已准备真实 provider 数值、LSE、分页 stride、动态长度 graph、实际 builder padding 和 eager 标量回读检测；本轮未执行 |

此前 runner 的 D2H/event wait 成对判断继续使用：只有所有 target 消费者支持设备长度时才跳过同步。
此次不扩大 PP、PCP、PD、量化或长上下文的支持范围。

## 长度安全约定

公开算子没有 `seqused_q` 参数。框架保证：Query 前缀以零开始、单调、最后一个前缀不超过 Q 的分配 token 数；
`used_q` 等于该段长度，或者整段为零；有效 KV 不超过分配容量。

生产 builder 从 Query 前缀之差计算 `used_q`，只会整段屏蔽无效请求，额外 padding 请求负责剩余 token。
mock 保留值断言供回归测试。native 热路径只检查 shape、dtype、device 等静态信息，不把设备上的长度读回 Host。
此约定不允许外部调用者传入“分配一段但仅使用部分 Query”的输入。

## 算子侧配套补丁

原分支在 TND 前向的 `PackedCuLast` 中执行 `cu[-1].item<int32_t>()`。这发生在选择 metadata 路径之前，
会使 eager 模式发生标量 D2H；只修改框架不足以消除它。

[a3_mla_device_lengths.patch](a3_mla_device_lengths.patch) 是针对上述固定 commit 的配套补丁：

- 对显式提供 scheduler metadata 的 TND Q 路径，依赖已有 metadata 调用方约定，不再读取 Q 最后一个前缀到 Host。
- 未提供 scheduler metadata 的路径继续保留原来的 Host 内容检查。
- 在接口注释中明确前缀合法性和容量是调用方前置条件，不能把补丁视为可以接受任意输入。

补丁不改变 attention 公式、tiling 算法、公开函数签名，也不修改全局安装的算子包。
它在隔离目录编译通过，但未做 NPU 执行验证，需要算子团队评审并完成硬件验收后收敛到正式版本。
**未应用该补丁的原发布分支仍有 eager 回读；不能把它标为完整的无 Host 等待路径。**

## 安装与配置

仓库自带 [安装说明](https://github.com/defeng2008/flash-attention-npu/blob/cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c/README.zh.md#准备)。
下列命令是安装指导，本轮仅执行了独立目录的 `build_ext --inplace` 编译，未覆盖正在运行服务的包。

```bash
git clone --branch mla-release-261116 --single-branch \
  https://github.com/defeng2008/flash-attention-npu.git
cd flash-attention-npu
git checkout cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c
# 910 v3 的构建只需 CATLASS 本体，不需要递归拉取 DSL 的 LLVM 子模块。
git submodule update --init csrc/catlass
git apply /path/to/vllm-ascend/docs/source/developer_guide/Design_Documents/a3_mla_device_lengths.patch

source /usr/local/Ascend/cann/set_env.sh
FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_JOBS=4 \
  python -m pip install -e . --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
```

与 Python/PyTorch 匹配的 torch_npu、setuptools>=70.1、packaging、CANN 和 HCC 工具链需预先就绪。
本轮编译环境为 Python 3.12.16、torch 2.10.0+cpu、torch_npu 2.10.0.post7、CANN 9.3.0、Linux aarch64。

vllm-ascend 的 additional config 使用：

```json
{
  "a3_mla_sink_mode": "external",
  "a3_mla_sink_module": "flash_attn_npu_3.flash_attn_npu_interface",
  "a3_mla_sink_allow_graph": false
}
```

模型或 draft 需 enforce_eager。graph 验证通过前不默认放开。临时配置保留用于联调和回退，
待完整模型验收后再收敛为正式能力选择，不能因为算子可以导入就自动启用。

当前范围：K3.1 DSpark、V2 runner、A3、FP16/BF16、未量化 ND、64/512、单 KV head、PP=PCP=1，
max_model_len<=16384；1536 管理块仍转换为 128 算子页。更长上下文需独立验证后扩展。

## 验证结果

| 项目 | 结果 |
| --- | --- |
| Attention/A3/A5/DCP 等 CPU 回归 | 159 passed |
| runner / PP CPU 同步策略回归 | 92 passed |
| 固定发布分支真实 Python wrapper 的 CPU 接口探测 | 20 passed；底层 extension 为 stub，不能当作 NPU 数值结果 |
| 原始发布分支 v3/910 构建 | 编译和链接通过 |
| 发布分支加配套补丁 v3/910 构建 | 编译和链接通过 |
| 实际 attention kernel、graph、模型、HCCL、性能 | 未运行，遵守用户停止 NPU 验证的要求 |

用户停止前，仅完成一个 `torch._assert_async` 的小型设备能力探测，发现 CPU fallback；
没有启动新增的真实 attention 测试。该探测不计入算子接入通过项。

CPU 回归命令：

```bash
python -m pytest --confcutdir=tests/ut/attention \
  tests/ut/attention/test_a3_mla_sink.py \
  tests/ut/attention/test_flashmla_dcp.py \
  tests/ut/attention/test_flashmla_prolog_v3_cpu.py \
  tests/ut/attention/test_mla_prolog_v3_tile_size_cpu.py -q
python -m pytest --noconftest tests/ut/worker/test_spec_pp_cpu_counts.py -q
```

后续资源可用且允许 NPU 验证时，测试入口为
`tests/e2e/pull_request/one_card/test_a3_mla_sink_native.py`，可用 `--noconftest` 隔离运行。
其中 eager 标量读取检查在未修复的发布分支上应暴露上述回读；其他用例使用真实 external provider 和独立 FP64 reference。

完整验收还需：真实 K3.1 DSpark、请求重排/接受拒绝、混合 prefill、KDA 状态回滚、DCP 多 rank、图分支变化、
目标消费者同步判断，以及 profiler/同负载性能对照。当前代码继续以 Draft 评审。
