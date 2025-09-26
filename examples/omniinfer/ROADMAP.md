| Omni-Infer版本  | v0.4.0 | v0.5.0 | v0.6.0 |
|----------|----------|----------|----------|
| Performance & Reliability Optimization   | Chunk prefill支持PD分离【5】<br>APC亲和调度【4】<br>W4A8C16量化【7】<br>精度量化工具【7】 | 支持SGL推理框架【2】<br>Chunk Prefill混部入图【5】<br>MTP多头多token投机【6】<br>Eagle适配【6】<br>xPyD P侧动态扩缩容【2】<br>W4A8C8 【7】<br>支持MoonCake Store【1】 | xPyD D侧动态扩缩容【2】<br>Omni-attention支持Prefill加速【5】<br>故障快恢【3】<br>AFD架构设计 |
| Expanded Model Support | DS V3/R1<br>Kimi K2<br>Qwen3-235B-A22B | Qwen3-32B（BF16格式）<br>QwQ-32B （BF16格式） | GPT-OSS 120B<br>GPT-OSS 20B<br>DS V3.1<br>Qwen-VL |
| Usability Improvements   | Metrics信息与vLLM接口一致【9】  | PD分离自动部署（裸机、K8S）【2】<br>环境仿真工具Mock【9】 | 精度分析工具【7】<br>性能仿真模拟工具【9】 |

### Note
[1] SIG Cache Optimization<br>
[2] SIG PD Disaggregation<br>
[3] SIG EP Placement<br>
[4] SIG Scheduling SDK: **APC相关的[ISSUE](https://gitee.com/omniai/omniinfer/issues/ICO71W)和[PR](https://gitee.com/omniai/omniinfer/pulls/442)**<br>
[5] SIG Attention<br>
[6] SIG MTP<br>
[7] SIG Quant<br>
[8] SIG Ops and Graph<br>
[9] SIG Tooling and Testing<br>