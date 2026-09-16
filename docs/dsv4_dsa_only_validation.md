# DSV4 DSA-only 拆分（7.8 分支 A）

基点为 `3025c7133`，分支 `feat/dsv4-prefill-dsa-only`。
保留该提交的权重布局、Meta 和数值修复，不回退到含精度回归的 PR 状态。

新开关：`VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL_DSA=1`。
旧 `VLLM_ASCEND_FXRT_DECOMPOSE_DSV4_PREFILL` 不再作为别名生效。
仅纯 prefill/direct FX、无 ACL graph 模式允许拆分，decode 仍保留原算子。

| DSA 开关 | attention 图边界 | MoE 图边界 |
| --- | --- | --- |
| 0 / 未设置 | `vllm::dsa_forward` | `vllm::moe_forward_shared` |
| 1 | 原 DSA tensor 实现展开 | `vllm::moe_forward_shared` |

MoE 不再调用 `enable_decomposed_forward`；MoE 专属的序列并行 context、
overlap/event、routing active_num 和生命周期提示不再被 DSA 开关改变。
六个修改过的 MoE 文件（experts_selector、fused_moe、fused_moe_0_23_0、
moe_comm_method、moe_mlp、token_dispatcher）完整恢复为拆分前 `d20ac15e7`
的内容。DSA 开关不会改变 MoE 的事件、流或序列并行 context。
其他文件保留 `3025c7133` 的权重布局及 dummy 隔离修复。

## 四卡验证

131 容器 `dsv4-audit77-precision-20260916`，DP2×TP2、EP4、DSA-CP，
FXRT `0.1.dev0+5a29416`、Torch `2.10.0+cpu`、torch_npu `2.10.0.post2`。
没有修改 site-packages。使用四层固定非零合成 W8A8 权重，经真实 wo_a loader
形成三维布局；seed=1024、temperature=0、严格确定性计算。
不是完整真实 checkpoint 精度验证，也不覆盖 A3 所有通信路径或多 token decode。

305/737/1297/2257 token 自然语言请求各重复两次。对比真正 eager 的旧实现基线
`loaded3d_base_2345`（保留同样的 routing-v2 扩展兼容和权重初始化）。
比较最后有效 token 的完整 4096 维 hidden 和 129280 维 logits，不只比较生成文字。

| case | 后端 | DSA 开关 | 图中 DSA/MoE 大算子数（每图） | 8条请求 hidden/logits |
| --- | --- | --- | --- | --- |
| `audit78_A_rollback_on_gm` | 捕图后 gm.forward | 1 | 0 / 4 | 逐元素一致，最大差0 |
| `audit78_A_rollback_off_fxrt` | 直接 FXRT | 0 | 4 / 4 | 逐元素一致，最大差0 |
| `audit78_A_rollback_on_fxrt` | 直接 FXRT | 1 | 0 / 4 | 逐元素一致，最大差0 |

每组8条 HTTP200，重复请求自身完全一致。图捕获使用 `fullgraph=True`；
关闭时4个图文件、开启时10个图文件均满足上述边界，不宣称整网只编译一次。
旧环境变量不作为别名，PD角色/ACL模式/DSA-MoE隔离的7项单测通过。

测试发现扩展可能由 device allocator 加载、未经过 `enable_custom_op()`，
因此原 RMSQuant Meta 修正未必注册。现于 CP/non-CP DSA 初始化时也注册修正，
保证构图前已知输出 INT8，避免 dummy 包装对 INT8 再次动态量化。
保留原 fused kernel，未更改运行时数学计算。

证据位于容器 `/workspace/dsv4/audit77/`：`logs/<case>/comparison.jsonl`、
`debug/**/fx_graphs/`、`results/` 和 `tensors/<case>/`。
审计提交 `a845777e3` 在本分支代码上仅增加环境变量控制的 hidden/logits 保存钩子；
vLLM 审计提交 `6cc232384` 提供固定非零 dummy 初始化和 gm.forward 对照后端。
本分支不包含这些测试钩子。

复现前确认四卡空闲，替换卡号；case 名必须新建，脚本不会覆盖旧证据：

```bash
DEVICES_RANK0=4,5 DEVICES_RANK1=2,3 \
bash /workspace/dsv4/audit77/run_ablate78.sh NEW_CASE \
  audit/78-dsa-rollback audit/78-gm fxrt 1 loaded3d_base_2345
# 最后的 1 改为 0 验证关闭拆分；fxrt 改为 gm 验证捕图后的 PyTorch 执行。
```

脚本结束时释放自身服务进程组，保留容器、日志和数值文件。
