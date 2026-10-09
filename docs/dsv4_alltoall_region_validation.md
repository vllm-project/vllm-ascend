# DSV4 routed-expert region（7.8 分支 B）

基于分支 A，恢复 MoE runner 的 `enable_decomposed_forward`。
DSA 与 MoE 开关相互独立，默认均为0。分支名 `feat/dsv4-prefill-moe-alltoall-region`。

| `_PREFILL_DSA` | `_PREFILL_MOE` | DSA | MoE |
| --- | --- | --- | --- |
| 0 | 0 | 原大算子 | 原大算子 |
| 1 | 0 | 展开 | 原大算子 |
| 0 | 1 | 原大算子 | 展开 |
| 1 | 1 | 展开 | 展开 |

完整变量名为 `VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_DSA` 和
`VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_MOE`；旧的无后缀开关不生效。
两者都要求纯prefill、direct FX模式、无ACL graph，需重启服务生效。
MoE关闭时不启用runner分解入口或新region，保留`vllm::moe_forward_shared`。

每个worker初始化时，无论开关是否开启，都将DSA/MOE合并打印为一条INFO日志，标记为
`[DSV4_PREFILL_PATH]`。`requested`为环境开关，`active`为最终生效状态，
`path`为`decomposed`或原始`vllm::dsa_forward`/`vllm::moe_forward_shared`。
同时打印prefill/direct FX/无ACL graph条件，便于解释请求开启但未生效的情况。
日志在模型构造前打印，不在forward内打印，不引入graph break。

```bash
grep -F '[DSV4_PREFILL_PATH]' prefill.log
```

AllToAll 内最小的形状闭合范围是：

```text
router/top-k → [preprocess → dispatch → expert MLP → combine] → finalize
                        fxrt_alltoall_routed_experts
shared expert ───────────────────────────────────────────────→ merge
```

仅包 preprocess 或 dispatch 会将 `sum(output_splits)` 的不定长输出暴露给图，
专家 MLP 仍需该行数。直到反向 AllToAll 和 unpermute 后才恢复输入 `[T,H]`。
新包装不再向图输出 worst-case capacity buffer，不填充额外专家 token。
fake 输出的 T/H 直接继承 hidden_states 的 SymInt，expert counts 长度为静态本地专家数。
权重/scale 均显式作为图输入，实际路由值不会成为 graph guard。

内部保留原 `dist.all_to_all_single(async_op=True)` 和每个 consumer 之前的
`handle.wait()`，不是换成另一个 collective。CPU split读取使用同步D2H，
保证 numpy/sum/分配消费的是完成的拷贝。输入 Tensor 不写入，所有被 resize 的
buffer 均为内部 dispatch/MLP 临时结果；因此 schema 使用 `mutates_args=()`。
不要把这些内部 `resize_(0)` 移到调用方输入或权重上。

范围当前为 DSV4 静态 W8A8、无 LoRA、无动态 EPLB；其他量化/scale-offset 模式
显式拒绝进入该包装，不静默改写算法。

测试专用 `VLLM_ASCEND_FXRT_TEST_A3_ALLTOALL=1` 只对 A2 且 EP>1、token数
超过真实 MC2 capacity 生效；调用 A3 非融合分支选择 ALLTOALL。
小于等于容量时仍为 A2 原路径，不在 A2 上伪造可运行的 A3 MC2 kernel。
不应在生产环境打开；默认0。

## 7.9 独立开关回归

131同一四卡DP2×TP2环境，直接FXRT、fullgraph=True；A2测试开关保持1，
四种组合均与同一路径的真正eager基线`audit78_B_alltoall_eager`对比。
305/737/1297/2257 token自然语言各重复两次，共32次请求，全部HTTP200。
每组16项跨版本完整hidden/logits比较全部逐元素一致，最大差0；重复与TP副本也一致。

| case | DSA | MOE | 图内DSA大算子 | 图内MoE大算子 | AllToAll region |
| --- | --- | --- | --- | --- | --- |
| `audit79_d0m0` | 0 | 0 | 4 | 4 | 0 |
| `audit79_d1m0` | 1 | 0 | 0 | 4 | 0 |
| `audit79_d0m1` | 0 | 1 | 4 | 0 | 4 |
| `audit79_d1m1` | 1 | 1 | 0 | 0 | 4 |

表内为长请求图；MoE展开时另有2个空metadata路径图无AllToAll region。
DSA关闭共4个图文件，开启共10个，仍有既有DSA metadata引起的重编。
本轮仅修改env/门控策略，没有修改算子计算、FXRT包或权重初始化。
审计提交`0dbb53b5c`在被测生产代码上只增加hidden/logits采集钩子；最终提交
相对被测代码仅有文档、门控函数docstring和ACL拒绝分支测试断言的补充。
13项门控单测覆盖四种组合、未配置时回退、PD角色和ACL拒绝、旧开关不生效。
测试权重为固定非零合成W8A8，不代表完整真实权重精度验证。

切换只需分别设置两个开关并重启服务。四组证据在容器
`/workspace/dsv4/audit77/logs/audit79_d*m*/`，日志包含本组开关值、提交和图；
`comparison.jsonl`为完整数值比较，`results/`、`tensors/`保留响应和张量。

## 7.8 已完成验证（独立开关前）

131 容器 `dsv4-audit77-precision-20260916`，四卡 DP2×TP2、EP4、DSA-CP，
Torch2.10.0+cpu / torch_npu2.10.0.post2 / FXRT0.1.dev0+5a29416。
FXRT 源码干净，没有修改 site-packages。测试为四层固定非零合成 W8A8，
真实 wo_a loader 布局；不是完整真实权重或 A3 硬件的精度证明。

| 检查 | 证据和结果 |
| --- | --- |
| A2选择边界 | `test_a3_alltoall_selector_boundary.py`：容量32/1024、恰好容量/超容量、EP禁用/单rank、A3原选择均通过 |
| 参数边界 | `test_alltoall_region_payload.py`：激活枚举经schema字符串往返，恢复原MLP枚举；权重保持显式Tensor输入 |
| 动态形状最小图 | 四进程Gloo，token数7/19/33，改变路由值、含零接收rank，fullgraph动态捕图仅编译1次，结果正确 |
| 真实NPU eager基线 | `audit78_B_alltoall_eager`，关闭拆分且enforce-eager；capacity32，长请求选择ALLTOALL；8次输出确定 |
| 真实NPU FXRT拆分 | `audit78_B_alltoall_fxrt_isolated`，fullgraph=True；8请求HTTP200，完整hidden/logits与同路径eager逐元素一致、最大差0 |

自然语言长度305/737/1297/2257，各重复两次，temperature0、seed1024、严格确定性。
最后有效token完整hidden4096维、logits129280维，TP副本一致且有限。
对照采用同样AllToAll路径的eager；旧AllGather与AllToAll自身有微小数值差异，
不能把通信算法不同的差异归因于新包装。

实际图中DSA和MoE大入口都已展开，长请求每层1个
`vllm_ascend.fxrt_alltoall_routed_experts.default`，其输出为
`bf16[(s72//2),4096]`。`sum(output_splits)`、变长分配、Work对象不在外层FX图。
runtime仍调用原`async_all_to_all`，默认event=None分支执行
`dist.all_to_all_single(async_op=True)`，再由原dispatch/combine调用`handle.wait()`。
没有改成`all_to_all_v`，也没有改成同步collective。

整网共10个图文件：8个长请求图各含4个region，另2个空metadata路径图没有region。
全网仍因DSA compressor metadata（例如num_compressed_tokens）改变而重编，
**不宣称整网单图复用**。独立region的符号shape/路由值变化复用由Gloo小用例验证。

生产代码测试对应审计提交`3c050727f`；vLLM为`6cc232384`，
前者只在生产代码上添加数值采集钩子，后者包含固定非零dummy初始化。
服务结束后已释放4张卡，保留容器与证据。

## 复现

先确认卡空闲，再在131容器执行；每次使用新case名字：

```bash
# 不拆分、真正eager，建立相同ALLTOALL算法的基线。
VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_MOE=0 \
DEVICES_RANK0=4,5 DEVICES_RANK1=2,3 VLLM_ASCEND_FXRT_TEST_A3_ALLTOALL=1 \
bash /workspace/dsv4/audit77/run_ablate78.sh NEW_BASE \
  audit/79-independent audit/78-gm eager 0

# 直接FXRT，不经过Inductor；两个开关分别展开DSA与MoE。
VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_MOE=1 \
DEVICES_RANK0=4,5 DEVICES_RANK1=2,3 VLLM_ASCEND_FXRT_TEST_A3_ALLTOALL=1 \
bash /workspace/dsv4/audit77/run_ablate78.sh NEW_FXRT \
  audit/79-independent audit/78-gm fxrt 1 NEW_BASE

# 仅CPU/Gloo边界测试（不占NPU），在源码根目录执行。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --standalone --nproc-per-node=4 \
  tests/e2e/test_alltoall_region_boundary.py
```

`run_ablate78.sh`有排他锁、端口检查、精度断言和退出清理，不允许重叠运行。
证据在`/workspace/dsv4/audit77/logs/<case>/`、`results/`、`tensors/<case>/`。
早期`audit78_B_alltoall_fxrt_activation`和`audit78_B_alltoall_fxrt_final`请求重叠，
已明确排除，不作为精度证据；有效结果仅使用`isolated`组。

新增算子/测试的Ruff检查、格式检查、针对CRLF原文件的diff检查通过。
尝试执行`bash format.sh ci`，因宿主机缺少pre-commit退出，未完成全仓hook。
