# 统一构建验证设计

目标：一份原生二进制服务14B/27B/30B，保留原已验证库和R15及默认安装不变。采用R15完整源码的通用Dense/state回退与专用MoE/raw Cube分支，不将两份含同名Torch注册的.so直接链接。

14B/27B使用原验证环境（仅开关），共享R15通用分支是否等价Stage B必须通过逐元素和完整模型实测确认，不能因公共接口相同就宣称源码相同。30B使用完全相同R15参数/分块。两个profile只改变优化开关，三个模型使用同一native路径和SHA。

数学、dtype、tensor签名沿R15 README/ops.h，group128，single/multi FP16/BF16及state FP32均保持。无新增数值计算或近似。UB/L1/L0与原代码完全相同：raw Cube L1 81920、L0A16384、L0B65536、L0C4096；通用向量路径沿各kernel查询及原tiling，不新增buffer。Workspace沿原实现。

新增CPU metadata算子返回统一构建标识，不参与计算。现有single/multi/state/MoE注册不变。工程独立在csrc/ops/splitquant_dense_fused_merged，不安装wheel，通过环境变量加载。

检查：原库SHA不变；至少30项跨进程逐元素对照；27B真实page stride state I/O；三个模型配对vLLM性能；完整ARC_c1172/ARC_e2376/PIQA1838。30B原R15同时补完整数据集作对照。此任务验收是合并等价，不受此前“超过BF16才跑全量”的加速试验门禁约束；不伪造速度通过标记。
