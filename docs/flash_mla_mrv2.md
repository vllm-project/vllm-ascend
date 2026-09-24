# MRV2 外部 FlashMLA：NPU 联调候选

本候选将外部 FlashMLA 两段算子接到 MRV2 target 和 MLA DSpark draft，
同时保留非连续 BBND KV cache。**本地检查通过不等于 NPU、DSpark 或图模式已跑通。**

## 版本与 PR 依赖

本 Draft 直接接在 [非连续缓存 #16456](https://github.com/vllm-project/vllm-ascend/pull/16456)
的固定提交 `a583897e` 后，只展示 FlashMLA 接入增量。接入底稿来自
[#17124](https://github.com/vllm-project/vllm-ascend/pull/17124) 的 `2ec47279`；
移植时保留最新 #16456 的 cache view/helper、allocator、COW 和清零实现，
不把旧 #16456 代码或历史交接分支整包搬入。等待 #16456 合入后，
再对更新后的主线整理、复验算子增量；本 Draft 不作为已验收版本合并。

| 角色 | 固定 SHA |
| --- | --- |
| #16456 / 本 Draft 直接基线 | `a583897e0c67d9c23288686728124b04b511a462` |
| 已合入的 MRV2 DSpark #16347 | `d4511510e15eb52744089e40d8cd7c9fc23ac0f4` |
| #17124 接入底稿 | `2ec47279b60f674d4dcd43a7b8becd190434b8bd` |
| 配套 vLLM | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` |

#16456 基线已包含 #16347，不重复合入 DSpark。
机器可读记录在 [flash_mla_versions.json](flash_mla_versions.json)。

## 算子入口和保持的语义

- 开关：`VLLM_USE_V2_MODEL_RUNNER=1`、`VLLM_ASCEND_ENABLE_FLASH_MLA=1`。
  后一个开关默认关闭。
- 两段调用均从 `cann_ops_transformer.ops` 导入：先执行
  `flash_mla_with_kvcache_metadata` 生成调度，再由
  `flash_mla_with_kvcache` 消费。此适配层不调用
  `torch.ops._C_ascend.flash_mla_with_kvcache*`，也不在缺包时回退 FIA。
  其他 Ascend 算子仍可使用仓内扩展。
- 两段 `max_seqlen_q/max_seqlen_kv` 均为 `-1/-1`；内部正数容量仅用于分配 buffer。
  主算子布局固定为 `TND / PA_BBND / NTD`，QK/V 维度为 576/512。
- 保留 #16456 原始 `[P,128,1,576]` 缓存及 stride、storage offset、页表、slot、
  allocator、COW、清零语义。KV 写入使用原缓存的 512/64 切片，不整份 contiguous/repack。
- 实际 TP local Q heads 必须属于 `8/12/64/96`，不复制全头或伪造 metadata 参数。
  target 和 draft 必须分别满足这个条件。
- 真实长度读取设备上的接受/拒绝修正结果；显式传入 query 边界和有效长度。
  causal 使用 mask mode 3 及 int8 上三角 mask，noncausal 使用 mode 0、无 mask。
  物理 padding 使用零有效长度行，slot 为 -1。
- target/draft 各自持有调度和缓存；图外刷新调度，保持捕获 buffer 地址，等待生产完成，
  消费结束后才复用。MLA draft FULL replay 不重复提交调度，也不走 FIA 图更新。
- 保留 V-up、gate、O-proj，以及带条件限制的 BF16 gate/GEMM 路径。
  非 MLA 层继续使用原后端。
- DSpark 沿用当前主线 `load_draft_model` 和 `post_process` 对齐流程；
  不导入历史加载期旋转，避免重复旋转同一权重。

本轮范围为 A5、MRV2、dense MLA、BF16 KV、PCP/DCP=1、PP=1、无 PD/KVPP。
DCP 辅助代码仅用于后续准备，运行配置仍禁止 DCP>1。
这不是 C8、PA_Nz 或融合 prolog 的完整接入，也没有性能等同声明。

## 验证机准备

1. 拉取本 Draft 的 head，记录 `git rev-parse HEAD`，安装该 checkout；
   vLLM 使用上表的固定版本。不要用 `PYTHONPATH` 临时覆盖另一份已安装源码。
2. 准备兼容的 A5/CANN/torch/torch_npu/Triton，以及提供上述两个 Python 入口和
   Meta 实现的 `cann_ops_transformer` 算子包。此仓库不附带算子包安装文件或指定未经确认的版本。
3. 对照安装包的 wrapper/schema 和算子说明，确认 `-1/-1`、非连续 BBND 的 stride/offset、
   mask 0/3、NTD 输出及 Meta 调度大小。仅 import 成功不能证明这些约定成立。
4. 检查两个 checkpoint 的真实 heads、正常 TP 切分、draft block size、量化参数和内存容量。
   TP4 是参考配置，不是这台机器已经验证过的配置。不要复制 heads 来凑四卡。
5. 在实际运行目录执行下面的脚本；模型路径、源码路径、Python 环境由验证机确定。

## 参考启动脚本

脚本：[examples/flash_mla_dspark.py](../examples/flash_mla_dspark.py)。
使用启动服务的同一个 Python 解释器，打印目标命令、已加载源码/算子包路径和版本，
并调用 metadata 的 Meta 实现检查 `-1/-1` 下的 buffer sizing。
缺包或 Meta 失败会终止，不回退仓内 FlashMLA。
`--dry-run` 只打印命令，不导入加速器包、不启动服务。
`--preflight-only` 只检查安装入口和 Meta，不启动服务、不执行 NPU 张量算子。
默认模式是 graph；eager 和 graph 使用同一套实现。

```bash
# 按验证机填写，不假定旧机器路径仍存在。
export ASCEND_SRC=/path/to/vllm-ascend
export TARGET_MODEL=/path/to/target-checkpoint
export MLA_DRAFT_MODEL=/path/to/mla-dspark-checkpoint
export DRAFT_TOKENS=5  # 示例；必须换成 checkpoint 对应值。

# 先查看命令，可以在没有 NPU 的机器执行。
python "$ASCEND_SRC/examples/flash_mla_dspark.py" \
  --target-model "$TARGET_MODEL" --draft-model "$MLA_DRAFT_MODEL" \
  --num-speculative-tokens "$DRAFT_TOKENS" --mode graph --dry-run

# 验证机上，检查算子包路径、两个入口和 Meta。
python "$ASCEND_SRC/examples/flash_mla_dspark.py" \
  --target-model "$TARGET_MODEL" --draft-model "$MLA_DRAFT_MODEL" \
  --num-speculative-tokens "$DRAFT_TOKENS" --preflight-only

# 第一步：target eager。
python "$ASCEND_SRC/examples/flash_mla_dspark.py" \
  --target-model "$TARGET_MODEL" --target-only --mode eager

# 第二步：target + MLA draft eager，先停止上一服务并保留日志。
python "$ASCEND_SRC/examples/flash_mla_dspark.py" \
  --target-model "$TARGET_MODEL" --draft-model "$MLA_DRAFT_MODEL" \
  --num-speculative-tokens "$DRAFT_TOKENS" --mode eager

# 第三步：同样的输入，target + MLA draft FULL_DECODE_ONLY 图模式。
python "$ASCEND_SRC/examples/flash_mla_dspark.py" \
  --target-model "$TARGET_MODEL" --draft-model "$MLA_DRAFT_MODEL" \
  --num-speculative-tokens "$DRAFT_TOKENS" --mode graph \
  --devices 0,1,2,3 --tp 4 --draft-tp 4 \
  --max-model-len 4096 --max-num-seqs 4
```

默认监听 `127.0.0.1:8000`，服务名 `flashmla-dspark`。
需要时使用 `--host`、`--port`、`--served-model-name` 修改。
通过末尾 `--` 传 checkpoint 所需的额外 vLLM 参数，例如量化配置；
这些参数不能覆盖脚本选择的模式、dtype 和拓扑。
脚本不负责安装依赖、停止旧进程或保证模型内存足够。
如果 K3 四卡需要权重量化配置，必须补齐该 checkpoint 的参数；KV 仍要求 BF16。

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"flashmla-dspark","prompt":"Explain attention briefly.","temperature":0,"max_tokens":32}'
```

返回 HTTP 200 和非空文本只是 smoke。继续核对 draft 接受/拒绝、输出正确性、cache 生命周期
和图回放；监听端口或单次请求成功不是最终验收。

## 本地结果与待验收项

原 #17124 底稿在 macOS arm64、Python 3.12、CPU torch 2.8.0 上曾运行：

```bash
python -m unittest \
  tests.ut.attention.test_flash_mla_host_contract \
  tests.ut.spec_decode.test_flash_mla_dspark_host_contract \
  tests.ut.worker.test_flash_mla_integration_host_contract -v
```

当时共 40 项 CPU/mock 检查：attention 17、DSpark 15、integration 8。
包括设备长度变化、padding/mask、target/draft 隔离、图外更新、稳定地址及复用等待、
context KV 写入、非 MLA 图路径和仅一次权重对齐。
新增一项在仓内同名 FlashMLA binding 存在时仍只调用外部包的测试，
同时检查缺少外部入口时 metadata 和主算子均报错。

当时另有旧 #16456 的 4 项生命周期测试通过隔离 CPU harness 运行。
这些均是移植前历史证据，**不能换绑到本 Draft 的新 SHA**。
本轮 Windows 工作环境缺少 torch/numpy，CPU/mock 用例在导入阶段无法运行；
待具备配套依赖的环境重跑。当前完成语法、Ruff lint、16 个增量 Python 文件的
format 检查及差异检查。剩余 `worker/v2/attn_utils.py` 的两处 format 提示位于
未修改的 #16456 cache-view 代码，基线 `a583897e` 原样也有；为保留依赖代码不在此重排。
最新 #16456 的生产 COW/清零代码及缓存 view/helper 均未改。
测试中的包、事件、RoPE、NPU scatter 和 Triton 操作使用替代实现；
CPU 清零测试仅核对元数据。不能据此证明 NPU 数值或真实图捕获正确。

| NPU 项目 | 首轮需要覆盖 | 状态 |
| --- | --- | --- |
| 外部算子接口 | 两段 -1/-1、Meta、TND/PA_BBND/NTD、真实 local heads 8/12/64/96 | 待验证 |
| 算子数值和缓存 | KV 0/1/127/128/129/257、长上下文、跨页、非零 offset、双轴 stride、mask 0/3、padding | 待验证 |
| target eager | RoPE/NoPE、V-up/gate/O-proj、prefix cache、COW 和回收清零 | 待验证 |
| MLA DSpark eager | 逐层 context slot、query metadata、接受/拒绝长度、多步请求 | 待验证 |
| 图模式 | 相同输入对比 eager、不同 bucket、padding、页表更新、重复 replay、稳定 buffer | 待验证 |
| K3 四卡 | 真实模型权重、正常 TP heads、内存、请求正确性与接受率 | 待验证 |
| DCP 扩展 | history/current 合并、拓扑、collective、DSpark 组合 | 后续，不在首轮范围 |

小规模算子数值对照可先用 `atol=rtol=0.02`，同时单独分析误差；
cache、slot 和保护区要求精确一致。真实 checkpoint、完整模型、实际 NPU kernel、
collective、图捕获/回放和性能均没有本机验收记录。
每次验证保存 candidate/vLLM SHA、算子包版本、CANN/torch_npu、拓扑、模型配置、命令和日志。
