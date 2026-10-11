# MegaMoe 共享专家融合（fc=3）Profiling 对比报告

- 日期：2026-10-11
- 环境：141.61.33.12 / 容器 `zt_1009` / Ascend950DT ×8（TP8+EP8）
- 模型：GLM-5.3-W4A4C8-MXFP4（routed experts W4A4 MXFP4 + shared experts W8A8 MXFP8，`routed_scaling_factor=2.5`）
- 关键配置：`--enforce-eager`、`enable_dsa_cp=false`、`enable_flashcomm1=false`、`enable_shared_expert_dp=true`
- 采集方式：`--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'` 挂载 `/start_profile`、`/stop_profile`；warmup 一次后 start_profile → 相同长 prompt（6360 字符，md5 `8eb787c6a8355baa2184f7f0b2914d31`）curl → stop_profile，msprof 导出 `op_summary`
- 对照组：
  - **fc=2**：`enable_fused_mc2=2`（MegaMoe，仅路由专家，共享专家走独立路径）
  - **fc=3**：`enable_fused_mc2=3`（共享专家融合进 MegaMoe 算子）

## 1. 总体指标（rank0 设备侧）

| 指标 | fc=2（未融合） | fc=3（融合） | 差异 |
|---|---:|---:|---:|
| 设备活动窗口（device span） | 69.59 s | 66.46 s | **-3.13 s (-4.5%)** |
| 算子总耗时（多流累加，可重叠） | 129.71 s | 33.56 s | **-96.15 s (-74%)** |
| 单次 prefill 请求 e2e（curl time_total） | 69.61 s | ≈66.5 s（与 device span 一致） | -4.5% |

> 说明：算子总耗时为所有流上算子 duration 累加，多流并行时大于 span；两组的差值主要反映"从外部 HCCL/小算子序列收敛为算子内执行"的重构幅度。

## 2. 关键算子耗时对比（TOP 差异项）

| 算子 | fc=2 次数 | fc=2 总耗时 | fc=2 平均 | fc=3 次数 | fc=3 总耗时 | fc=3 平均 | Δ总耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MegaMoe** | 19200 | 21959.5 ms | 1143.7 us | 19200 | 10288.6 ms | 535.9 us | **-11670.9 ms (-53.1%)** |
| **HcclLaunchAicpuKernel**（HCCL 通信） | 59648 | 60370.0 ms | 1012.0 us | 40448 | 4586.4 ms | 113.4 us | **-55783.6 ms (-92.4%)** |
| QuantBatchMatmulV3（量化 GEMM） | 65280 | 613.0 ms | 9.4 us | 26880 | 215.5 ms | 8.0 us | -397.5 ms (-64.8%) |
| DynamicMxQuant | 79872 | 275.1 ms | 3.4 us | 41472 | 146.0 ms | 3.5 us | -129.2 ms (-47.0%) |
| aclnnCat_ConcatD_ConcatD | 77568 | 240.1 ms | 3.1 us | 58368 | 189.8 ms | 3.3 us | -50.2 ms (-20.9%) |
| aclnnForeachAddListV2 | 19200 | 82.1 ms | 4.3 us | 0 | 0 | — | -82.1 ms（算子内化） |
| SwiGlu | 19968 | 44.7 ms | 2.2 us | 768 | 1.7 ms | 2.2 us | -43.0 ms（共享专家 SwiGlu 移入算子） |
| aclnnInplaceMuls_Muls | 19200 | 44.4 ms | 2.3 us | 0 | 0 | — | -44.4 ms |
| aclnnMean_ReduceMean | 0 | 0 | — | 19200 | 53.5 ms | 2.8 us | +53.5 ms（新增，融合输出归一化辅助） |
| aclnnAbs_Abs | 0 | 0 | — | 19200 | 53.5 ms 中一并统计 | 1.8 us | +34.4 ms（新增） |
| aclnnInplaceCopy_Cast | 31490 | 51.9 ms | 1.6 us | 50690 | 87.9 ms | 1.7 us | +36.0 ms（fc3 布局适配 cast 增多） |

**未受影响的算子（两组一致，验证对照有效性）**：

| 算子 | fc=2 总耗时 | fc=3 总耗时 | Δ |
|---|---:|---:|---:|
| KvQuantSparseFlashAttention | 1028.3 ms | 1027.4 ms | ≈0 |
| MlaPrologV3 | 519.0 ms | 519.1 ms | ≈0 |
| aclnnMatmul_MatMulV3Common | 294.8 ms | 301.3 ms | +6.5 ms |
| AddRmsNormBias | 128.6 ms | 129.3 ms | ≈0 |
| TransposeBatchMatMul | 125.6 ms | 122.9 ms | ≈0 |
| MoeGatingTopK | 55.8 ms | 55.3 ms | ≈0 |

## 3. 功能分组汇总

| 分组 | fc=2 次数 | fc=2 耗时 | fc=3 次数 | fc=3 耗时 | Δ |
|---|---:|---:|---:|---:|---:|
| MegaMoe（MoE 主算子） | 19200 | 21959.5 ms | 19200 | 10288.6 ms | -11670.9 ms |
| 通信（HCCL） | 59648 | 60370.0 ms | 40448 | 4586.4 ms | -55783.6 ms |
| 量化 GEMM（QuantBatchMatmulV3） | 65280 | 613.0 ms | 26880 | 215.5 ms | -397.5 ms |
| DynamicMxQuant | 79872 | 275.1 ms | 41472 | 146.0 ms | -129.2 ms |
| 注意力（Flash/KvSparse/MlaProlog/Indexer） | 56064 | 1611.5 ms | 56064 | 1614.1 ms | +2.6 ms |
| 归一化（RmsNorm/LayerNorm） | 45568 | 146.8 ms | 45568 | 147.7 ms | +0.8 ms |
| 数据搬运（Cast/Concat/Copy） | 109435 | 293.1 ms | 109435 | 278.9 ms | -14.2 ms |

## 4. 共享专家融合的功能性确认

### 4.1 算子输入证据（最直接）

对 MegaMoe 算子行的输入 dtype 字段检查：

- **fc=2**：输入仅含 `DT_FLOAT4_E2M1`（路由专家 MXFP4 权重），**无 E4M3**
- **fc=3**：输入同时含 `DT_FLOAT4_E2M1`（路由专家）与 **`DT_FLOAT8_E4M3FN`（共享专家 W8A8 权重）**，且权重布局为 FRACTAL_NZ 的分组数量增加

即 fc=3 模式下共享专家的 W8A8 权重作为独立分组直接传入 MegaMoe 算子。

### 4.2 算子数量此消彼长

- SwiGlu：19968 → 768（每个 MoE 层减少 1 次，与 MegaMoe 调用数 19200 完全对应；剩余 768 为 dense 层）
- QuantBatchMatmulV3：65280 → 26880（共享专家的 up/gate/down 三个 W8A8 GEMM 全部移入算子）
- DynamicMxQuant：79872 → 41472（共享专家侧的外部激活量化消失）

### 4.3 运行日志证据

- fc=3 日志中出现 `[MEGAMOE_AB] fold_rsf=2.5 layer_own_rsf=1.0 topk_sum=2.5000` 64 次（该日志仅在 `shared_w1` 权重传入 MegaMoe 时打印），`routed_scaling_factor` 正确折叠进 topk_weights
- `[MEGAMOE_DBG] fused_call=1 ... runner_rsf=2.5`，pre/post reduce ratio=1.000，输出缩放正确
- fc=2 日志中两者均为 0

## 5. 性能差异分析

1. **通信开销大幅收敛（最大收益来源）**：HCCL 类算子耗时 60.37 s → 4.59 s（-92%）。fc=2 模式下 MoE 前后需要独立的 dispatch/combine 通信（AICPU launch kernel duration 包含等待通信完成的时间），fc=3 融合后通信被 MegaMoe 算子内的对称内存路径吸收，外部 HCCL 仅剩 40448 次轻量调用（平均 113 us，较 fc2 的 1012 us 降 89%）。

2. **MegaMoe 单算子变化**：单次平均 1143.7 us → 535.9 us（-53%）。fc=2 的 MegaMoe 内部等待通信；fc=3 的执行更连续，且总耗时减少 11.67 s 的同时承担了共享专家计算，说明路由专家与共享专家的批内融合有效摊薄了通信与调度成本。

3. **外部小算子链路消除**：共享专家在 fc=2 下走独立路径（独立 GEMM + SwiGlu + 动态量化 + AddList 合加），融合后该链路整体消失（QuantBatchMatmulV3 -397 ms、DynamicMxQuant -129 ms、SwiGlu/ForeachAddListV2/InplaceMuls 合计约 -170 ms），仅新增 Mean/Abs 等轻量辅助算子（约 +88 ms），净收益约 -800 ms 量级（rank0 纯算子口径）。

4. **调度与流开销**：fc=3 的 Cast 次数增加（31490 → 50690，+36 ms），来自算子输入输出布局适配，属可接受代价。

5. **端到端收益**：长 prefill 请求 device span 69.59 s → 66.46 s（-4.5%）。本请求为单请求 prefill 场景，注意力/MLA 等占比不变，MoE+通信路径的收益即为全部收益；批量并发/decode 场景下 MoE 与通信占比更高，预期收益放大。

## 6. 结论

- fc=3 共享专家融合**真实生效**：算子输入 dtype、算子数量此消彼长、运行日志三层证据一致。
- 融合在保证数值正确（scale 折叠正确、输出校验通过）的前提下，rank0 设备侧算子总耗时下降 74%，端到端 prefill 提速约 4.5%（单请求 prefill 场景），主要收益来自 HCCL 通信收敛（-92%）与共享专家外部算子链路消除。

## 附录

- trace 目录：`/root/prof_fc3`（32G）、`/root/prof_fc2`（38G），各含 8 rank 的 ascend_pt 原始数据 + 前端 `*.pt.trace.json.gz`
- msprof 导出结果：`.../PROF_*/mindstudio_profiler_output/op_summary_*.csv`
- 汇总数据：容器内 `/root/pf_cmp_result.json`
- 采集脚本：容器内 `/root/serve_prof.sh`（FC_VALUE/PROF_DIR 参数化）、`/root/pcf3.sh`、`/root/pcf2.sh`
- 服务已停止，NPU 资源已释放（容器已重启）
