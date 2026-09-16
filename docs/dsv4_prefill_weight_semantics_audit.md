# DeepSeek V4 prefill 拆分数值审计（7.7.3）

审计 `19dcffcf601ad830d6c483188d1a2860efcad8c1`，修复基于
`27d02d8171c61f0450878045d7af9d46f4535073`。分支
`fix/dsv4-prefill-weight-semantics`；最终仅在该基点之后保留一个修复提交。

## 已确认的问题与修复

| 位置 | 拆分提交引入的变化 | 修复 |
| --- | --- | --- |
| `attention/context_parallel/dsa_cp.py` | 对真实 loader 已转换为 `[G,K,R]` 的 wo_a 再次 view/transpose；形状不变，元素排列改变 | 合并 `c077c53da`，仅转换原始二维权重，三维对象直接使用 |
| `attention/dsa_v1.py::_forward_o_proj` | 非 CP 路径存在相同的无条件重复转换 | 同样只转换二维权重 |
| `attention/dsa_v1.py::_forward_prefill` | 用 RMSNorm 再 DynamicQuant 替代 fused RMSQuant，引入 BF16/FP16 中间舍入 | 恢复原 fused 算子，单独修正 Meta 的 INT8/FP32 输出类型 |
| `attention/dsa_v1.py::_mla_prolog_multistream` | 拆分关闭 overlap 后改走另一套串行 prolog：默认多流 prefill 原本用独立 RMSNorm/quant 并给 indexer 传 BF16 qr，而串行分支使用融合 RMSQuant 和 INT8 qr | 保留原配置选择的 CV prolog，只禁用 stream/event；显式配置非多流时仍保留原融合算子路径 |
| `device/device_op.py::apply_dsa_q_rms` | 用 ATen FP32 表达式替代 Triton；FP16 实测输出不同 | 有 Triton 时保留原 kernel，拆分路径通过 `fxrt_dsa_q_rms` 自定义算子调用 |
| `attention/dsa_v1.py::forward` | 用传入 DSA 的 positions 长度替代全局 token 数；四卡305-token请求的 MTP attention 收到153行局部 positions，主模型为306行（含padding）。先在 RoPE 报 dim0 不一致，单独修复 prefill 截取后又在153行 buffer 写305行时失败；旧 eager 正常 | 分别从 common/prefill RoPE metadata 的 tensor 形状获取投影 buffer 行数和真实 prefill 行数 |
| `ops/dummy_quant_matmul.py`、`ops/fxrt_moe.py`、`ops/register_custom_ops.py` | 仅靠 DUMMY_QUANT 环境变量即可进入重复量化、DP token 截取或输出补零逻辑，未检查 loader | 必须同时满足 `load_format == "dummy"` 和环境变量为 1；真实 loader 即使误设变量也不进入 |

没有修改原始 weight_loader、NZ 转换、量化 scale 加载或实际 fused RMSQuant kernel。
三维 wo_a 不进行 copy、reshape 或 format cast，保留对象、storage、格式。
本轮没有修改 FXRT，也不需要重编 vllm-ascend C++ 扩展。

Meta 在扩展 schema 注册后通过 Python 注册；支持加载顺序下的重试与重复调用。
实际执行仍调用 `_C_ascend.npu_rms_norm_dynamic_quant`，不是 BF16→INT8 模拟。
Q RMS 自定义算子输出新 tensor，不修改输入，`mutates_args=()`；fake 保留形状与 dtype。

## 其余差异逐项核查

| 范围 | 结论及边界 |
| --- | --- |
| `ops/linear.py` 及 quantization loaders | 19dcffcf 未修改 loader；真实二维 checkpoint 经 loader 转成三维，与跳过 loader 的 dummy 路径不同，这是 wo_a 回归触发条件 |
| MoE routing `active_num=-1` | 与 `tokens*top_k` 的有效行、expanded indices、expert counts、量化 scales 单算子对照；按 expert counts 排除未初始化的尾部 buffer，非量化时不比较未使用的 scale |
| legacy routing custom→`npu_moe_init_routing_v2` | 确为无条件算子替换。当前扩展缺少旧算子，旧 eager 对照也需要相同 v2 兼容；不能以本轮结果宣称已验证旧 kernel 与新 kernel 所有配置等价 |
| indexer/scatter | helper 内 quant、scale cast 和 scatter 与原代码相同；cache 写入 custom op 已声明 `mutates_args`。空输出不写 cache；non-CP helper 返回的原始 kv/None 不被后续 QLI 使用，QLI 消费已更新的 cache |
| metadata 排序 | 按唯一字符串 key 排序与原 sorted(items) 的顺序一致，不改变对应 tensor |
| prefill 强制分支 | has_prefill、decode offset 与 full-gather 简化只适用于纯 P。生产门控未放开 eager；整网 eager 审计的放开门控是独立测试提交。不得把该路径用于 mixed P/D 或 decode |
| MTP hidden buffer | 不能笼统把 positions 当全局行数：采样钩子实测主模型为306行，MTP attention 为153行、位置0/2/.../304。修复非 CP DSA token 推导后覆盖该 draft prefill；只生成一个 token，不宣称验证了多 token MTP decode |
| MoE sequence context | 移除的 context 写入 `dp_metadata.local_sizes`；此版 Ascend DSV4 dispatcher 直接读取 `num_tokens_across_dp_cpu`。原“single-DP/no metadata”注释不准确，已改正；本次覆盖 DP2 |
| 多流/event | 分解时 MoE/DSA overlap 被关闭，MoE event 的 None 与串行执行配套；DSA 原 prolog 的中间值/舍入必须保留，不能把另一套串行计算直接视为等价。KV 通知/等待在 custom op 内执行。eager 对照不证明其他编译器的 side-effect/DCE 行为 |
| `dispose_tensor` | 拆分时跳过 `set_` 生命周期提示，避免破坏图内后继输入，不改变数学计算 |
| experts mask/where | 整数 token/expert id 上标量 -1/0 改为同 dtype tensor，语义相同。identity zero-expert 分支计算相同；不支持类型在原版已无 result，未扩大本轮功能范围 |
| 新/旧 vLLM runner | v0.23.0 路径使用对应文件；补齐另一 runner 新调用的缺失 import，避免其 NameError，但未声称其他 vLLM 版本通过整网测试 |
| `platform.py`、`__init__.py`、runner、fence | 入口选择、初始化、编译入口及事件发布，没有其他权重重排。测试/utils/platform 变动是门控测试。未发现第二处 loader 或 NZ weight 改写 |

## 单算子证据

在 131 / `dsv4-audit77-precision-20260916`，A2，torch 2.10.0+cpu、
torch_npu 2.10.0.post2、CANN 9.0.1，使用空闲的一张卡运行：

```bash
# 使用该容器已有 CANN/driver 环境，设置一张空闲卡的可见性。
export VLLM_VERSION=0.23.0
/workspace/dsv4/.venv/bin/python \
  /vllm-workspace/vllm-ascend/tests/e2e/pull_request/one_card/test_dsv4_prefill_weight_semantics.py
```

- CP/non-CP 从生产函数提取实际 layout 代码，groups=2/4/8，二维和已加载三维权重，真实 NPU batchmatmul 对照；同时断言旧错误变换确实不同。
- DUMMY_QUANT=1 时 auto/safetensors/dummy/auto 切换，核对真实输入对象不变、仅 dummy 量化；INT8 输入不重复量化；env=0 不启用。
- DP 输出回填与 hash token 截取使用生产函数和 mock 通信/kernel，核对真实 loader 不进入 dummy 特判；不是分布式 kernel 精度测试。
- RMSQuant 使用 BF16/FP16、256/257 行，实际输出、Meta、dynamic fullgraph gm.forward 对照。
- Q RMS 使用 BF16/FP16、8/32/257 行，原 Triton 与新包装 fullgraph 对照。
- Routing 使用8/257行、quant_mode=-1/1、部分 expert range，对比所有有效输出。
- 非 CP token 截取和投影 buffer 使用生产分支代码，对比305/306/153、737/738/369、256/256/128、1/32/16的有效/全局/TP局部行数；不把局部 positions 长度当全局 token 数。
- 生产 CV prolog 提取后比较串行/多流 Q、qr、KV 及 dynamic fullgraph gm.forward：使用真实 NPU quant/RMS，线性权重和 RoPE/cache sink 为简化替身；断言串行路径不取 stream、不发 event，且 qr 保留 BF16。整网另使用实际算子验证。

修复前独立探针：RMSQuant 两算子替换在256/257行的反量化最大差约0.07～0.09；
原 fused kernel 重复执行一致。另在512 hidden、8/32行观察到原 fused kernel 自身不稳定，
因此没有拿这些形状作为“拆分引入”的证据，也没有在此修改厂商 kernel。
Q RMS 的 FP16 原/新表达式最大差0.00390625，BF16 本轮形状逐元素相同。

## 四卡整网与验证边界

此前 CP wo_a 单修复 `c077c53da` 已完成 DP2×TP2、EP4、DSA-CP 真 eager 三方对照：
固定非零合成权重通过真实 wo_a loader 转为三维，305/737/1297/2257 token
自然语言各重复两次。未修复版 logits 最大差0.195～0.220；修复版完整 hidden/logits
及首 token 与旧 eager 基线逐元素一致。重复请求自身也完全一致。

整网使用四层非零合成权重，不是实际 checkpoint；基线保留必要 routing v2 兼容及数值保存钩子。
A3/A5、真实完整权重、MC2/AllToAll 分支穷举、多 token decode 不属于已完成验证范围。
这些限制不应被“HTTP 200”或单次首 token 相等掩盖。

本轮合并修复完成四卡回归：DP0用4/5卡、DP1用2/3卡，每种配置四条自然语言
请求各重复两次，8条均 HTTP200；固定非零 dummy 权重、真实 wo_a loader、
seed1024、temperature0、严格确定性。两侧20条 wo_a 加载记录完全一致。

| 路径 | 基线 case | 修复 case / 审计提交 | 完整 hidden/logits |
| --- | --- | --- | --- |
| DSA-CP=true | `loaded3d_base_2345` | `audit773_fixed_2345` / `ec78f1328` | 8/8逐元素一致，最大差0 |
| DSA-CP=false | `audit773_noncp_base` | `audit773_noncp_fixed_cv` / `c08b5d3be` | 8/8逐元素一致，最大差0 |

CP 测试之后的生产代码变化仅在非 CP 的 `dsa_v1.py`；CP 实现不是该类的子类，
本轮后续非 CP 修复不改变已测 CP 函数。两组最后位置 hidden 为4096维，logits为129280维；
自身重复请求、TP副本、API首token/logprob均完全一致且最终值有限。

非 CP 反例 `audit773_noncp_fixed_global` 已修复 wo_a/RMS/行数，尚未保留 CV prolog，
8条请求虽全部成功且首token相同，但 logits 最大差依次为
0.0203552/0.0136032/0.0134888/0.0122070。仅补上 CV prolog 序列化修复后全部归零。
因此不能把关闭多流后的另一套数学实现视为天然等价，也不能只检查首token。

另用相同采样钩子重新运行旧基线 `audit773_stage_base`，与无钩子基线结果完全相同。
305-token请求的90项 module边界张量记录覆盖各层 attention、MoE、layer 以及 MTP
prefill，**有效token行**全部逐元素一致且有限。306行全局buffer的最后一行、
TP1的153行局部buffer最后一行是padding，两侧均可含未初始化NaN；原始比较仍保留，
没有将padding行误算为有效数值差异。最终选取位置的 logits/hidden 没有NaN。

证据目录：127 `/home/liyizhan/dsv4/audit77/`；
131容器 `/workspace/dsv4/audit77/`。`audit773_final_units.log` 为最终单算子结果，
`stage_comparison_valid.jsonl` 为分层有效行对照，四卡结果在各 case 的
`logs/*/comparison.jsonl`、`results/`、`tensors/`；
归档 `audit773_evidence_20260917.tar.gz` 保留上述日志、原始张量和复现脚本。

### 复现入口与代码隔离

`run_ablate.sh` 每组启动两个 API/DP 进程，结束时释放其进程组，不删除容器。
`server_audit.sh` 的 `DSA_CP` 参数仅用于区分 CP/non-CP；默认 true。
其余 DP2×TP2、EP4、固定 seed=1024、确定性、相同权重初始化保持一致。

```bash
export DEVICES_RANK0=4,5 DEVICES_RANK1=2,3  # 先确认这四张卡空闲
bash /workspace/dsv4/audit77/run_ablate.sh audit773_fixed_2345 \
  audit/773-eager-candidate audit/77-eager-split eager 1 loaded3d_base_2345
DSA_CP=false bash /workspace/dsv4/audit77/run_ablate.sh audit773_noncp_base \
  audit/77-base-loaded3d audit/77-eager-base eager 0
DSA_CP=false bash /workspace/dsv4/audit77/run_ablate.sh audit773_noncp_fixed_cv \
  audit/773-eager-candidate audit/77-eager-split eager 1 audit773_noncp_base
```

已有 case 不可覆盖，重跑请换 case 名并同步修改基线名。
审计分支额外允许 enforce-eager 拆分、采集 hidden/logits、打印一次实际拆分入口；
正式修复分支不包含这些钩子，也不包含非零 dummy 初始化补丁。

### 静态检查

新增文件和修改文件的定向 Ruff 检查通过；两处原有文件的 E402 / I001 / F401
已与 `27d02d817` 原文件对比，均为基点既有问题，没有扩大无关格式变更。
`git diff --check` 通过。尝试执行 `bash format.sh ci`，因宿主机缺少
`pre-commit` 未执行完整仓库 hook，因此不宣称全仓 CI 通过。
