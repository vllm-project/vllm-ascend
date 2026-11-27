# HY4 Flash 昇腾适配

文本和视觉模型使用 Model Runner V2，支持 FP8 权重、BF16 计算、FP8 KV、
DFly 投机解码和 Mooncake PD 分离。部署依赖、配置与代码布局见
[集成说明](../../docs/source/developer_guide/hy4_decode_integration.md)。

## 当前运行路径

- Prefill 支持分页 KV、prefix cache、Engram CPU offload，以及 eager TP2/PP2。
- Decode 支持 TP1/DP4 外部负载均衡、目标模型 `FULL_DECODE_ONLY` 图回放，
  以及 `hy4flash_dfly_dp_graph` 控制的小模型入图。
- 三个可选优化是分页 IndexPool、C8 sparse attention 和 C8 SWA attention；
  默认关闭，条件与验证范围见 [C8 Decode 说明](../../docs/source/developer_guide/hy4_decode_c8.md)。
- 优化条件不满足时保留回退；FP32 KV、learned sink、MoE 和 iHC 精度边界保持不变。

## 启动入口

- `start_hy4flash.sh`：单服务示例。
- `serve_decode_external_dp.sh`：通过八个位置参数启动外部 DP Decode rank。
- `pd_118/`：P TP4 与 D TP1/DP4 示例；使用前修改本机路径与通信资源。
- `all-optimizations.json`：三个 CANN Decode 优化的组合配置。

精度和性能测试使用 AISBench，保留原始 result 文件。历史版本的数据不作为
当前 PR 的性能结论；重构后需在固定版本和相同配置下重新评测。
