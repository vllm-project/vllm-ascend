# K3.1 A3 MLA 算子接入

本方案在 [Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876) 中实现。
算子版本固定为 [mla-release-261116 / cc3d5ca](https://github.com/defeng2008/flash-attention-npu/tree/cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c)。

## 目的与选择规则

DSpark 的接受/拒绝结果先产生在设备上。下一轮 attention 如果要求 CPU 上的准确长度，
框架就必须 D2H 拷贝并等待。让 metadata 和 attention 直接消费设备长度，才能去掉这条依赖。
只替换 attention 算子还不够：target 的 SFA、Indexer、KDA 和状态管理也必须不依赖准确 CPU 长度。

A3 与 A5 一样，在初始化、分配缓存之前按能力选择，不需要新增 additional config：

- A3、K3.1 DSpark draft、V2 runner、单 KV head、内容 512 / RoPE 64、FP16/BF16、未量化 ND。
- 当前 PP=PCP=1，未配置 PD connector；管理块大小必须是 128 的整数倍。
- 固定导入 `flash_attn_npu_3.flash_attn_npu_interface`，需要两个公开函数同时存在。
- 缺包、缺接口或条件不符时保留 FIA；已安装包内部的依赖错误、实际算子执行错误直接抛出。
- eager / graph 使用框架原有执行配置。按 block table 容量传 KV 上界，不另外设置 16K 限制。

自动选择表示框架调用路径已实现，**不表示 graph、长上下文或完整模型已通过硬件验收**。

## 如何参考 A5

| 环节 | A5 已有实现 | A3 接法 |
| --- | --- | --- |
| 选择入口 | `AscendMLAImpl._can_use_flashmla` 检查硬件、形状和算子包 | 同一初始化位置检查 A3 / K3.1 条件和真实接口 |
| 长度准备 | `FlashMLAMetadataBuilder` 从设备 seq_lens / query_start_loc 刷新 | 共用同一个 builder，沿用 padding、请求重排和 DCP 长度计算 |
| metadata | `flash_mla_metadata`；Meta 查询缓冲区容量 | `get_scheduler_metadata`；用设备零长度查询，保存原生 uint8 缓冲区 |
| attention | `flash_mla`，融合 576 维 Query / KV | `flash_attn_with_kvcache`，Query 拆成 q(64) / qv(512)，KV 为独立 component |
| 图执行 | 固定地址，逐轮刷新长度和 schedule | 共用 builder 的分配、缓存和 `copy_` 刷新逻辑 |
| DCP | 历史/当前计算、跨流依赖、LSE 合并、V-up | 共用原流程；A3 current KV 的两块存储均记录 stream 生命周期 |
| runner 同步 | draft 和 target 都可消费设备长度时跳过 D2H / wait | 按实际初始化的 target 消费者能力判断；未知消费者仍同步 |
| prefill | 真正的 prefill 仍走 FIA，混合批次分开处理 | 共用同一分流，不把单 token prefill 当作 decode |

代码调用链：

```text
mla_v1.py: 初始化选择 + AscendMLAMetadataBuilder
  → flashmla.py: FlashMLAMetadataBuilder.build（逐轮刷新）
  → a3_mla_sink.py: build_metadata → get_scheduler_metadata
  → mla_v1.py / context_parallel/mla_cp.py: _forward_external_flashmla
  → a3_mla_sink.py: attention → flash_attn_with_kvcache
  → 原有 DCP LSE 合并 / V-up
```

`worker/v2/spec_decode/dspark/speculator.py` 不再为已选中的 A3 路径准备准确 CPU 长度。
`worker/v2/model_runner.py` 用同一个判定同时控制 D2H 提交和 event 等待，避免只删除其中一端。
SFA / Indexer / KDA / hybrid state 的能力声明用于保护这一步；PP、PCP、offload、adaptive、
PD 和未声明的消费者继续保留同步。

## A3 接口约定

- `cache_seqlens` 为 NPU int32 `[B]`，`cu_seqlens_q` 为 NPU int32 `[B+1]`。
- Query 前缀从 0 开始、单调，最后一个前缀不超过分配的 token 数。有效 KV 不超过 page table 容量。
- `used_q` 由 builder 用前缀差计算，或整段置零；公开算子不支持部分使用一段 Query。
- 128-token 算子页与 1536-token 等管理块不同，沿用现有缓存视图转换。
- KV component 在页内连续，允许页间 stride；不接受融合 576 维缓存后临时切片转连续的兼容路径。
- 显式传模型的 `softmax_scale`。metadata 保持原生连续 uint8，不解释其内部字段。
- 算子返回 `(out, lse, out_accum, lse_accum)`，adapter 将 TND 输出转为框架需要的 NTD。
- 对无效请求/padding，将 KV 长度置零，并把对应输出置零、LSE 置为负无穷。

热路径只做 shape / dtype / device 等静态检查，不读回设备长度，也不使用会回退到 CPU 的长度值断言。

## 算子侧仍需合入的补丁

上述固定版本在 TND 前向的 `PackedCuLast` 中执行 `cu[-1].item<int32_t>()`，
发生在选择 metadata 路径之前，会在 eager 模式引入标量 D2H。
[a3_mla_device_lengths.patch](a3_mla_device_lengths.patch) 为显式 metadata 路径去掉此 Host 内容检查，
同时声明前缀合法性由调用方保证；未提供 metadata 的路径保留原检查。

此补丁**没有新增设备侧合法性校验**，也不改变 attention 公式或函数签名。
需要算子团队评审这个调用方约定，并将修改收敛进正式发布版本；届时删除配套补丁。
未应用补丁的原始算子仍有 eager 回读，不能宣称已经完全消除 Host 等待。

## 安装

参考算子仓库的 [安装说明](https://github.com/defeng2008/flash-attention-npu/blob/cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c/README.zh.md#准备)。
以下仅为安装指引，不会自动修改正在运行的环境：

```bash
git clone --branch mla-release-261116 --single-branch \
  https://github.com/defeng2008/flash-attention-npu.git
cd flash-attention-npu
git checkout cc3d5ca2a85d9b5dc2417ebbcabe9d40fdf0c55c
git submodule update --init csrc/catlass
git apply /path/to/vllm-ascend/docs/source/developer_guide/Design_Documents/a3_mla_device_lengths.patch
source /usr/local/Ascend/cann/set_env.sh
FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_JOBS=4 \
  python -m pip install -e . --no-build-isolation --no-deps \
  --config-settings editable_mode=compat
```

安装后重启 worker，使初始化时重新选择后端；日志中的 `A3 MLA sink` 表示选择了此路径。

## 验证边界

当前 CPU 检查覆盖选择/回退、缓存布局、长度屏蔽、target 能力判断、D2H 与等待成对控制，
并运行原有 A5 的 DCP、prolog 和 runner / PP 回归。具体结果记录在 PR 中。
原始与打补丁的指定算子源码已在隔离目录编译、链接通过，未覆盖服务中的算子安装。

用户要求不使用当前占用中的 NPU，因此本轮没有执行 NPU 验证。
真实算子测试入口为 `tests/e2e/pull_request/one_card/test_a3_mla_sink_native.py`：
覆盖 FP16/BF16、causal/noncausal、分页 stride、padding、显式 scale、动态长度 graph、
生产 builder 和 eager 标量回读检测。CPU 检查不替代这些测试。

合入前还需验证：完整 K3.1 DSpark 接受/拒绝及重排、KDA 状态回滚、混合 prefill、DCP/HCCL 多 rank、
长上下文、图分支变化、实际同步策略和相同负载下的 profiler / 性能对照。PR 保持 Draft。
