# PD Balancer 策略

## 概述

PD Balancer 是 vLLM Ascend 新增的 EPLB（Expert Parallel Load Balancing）策略，策略类型编号为 **4**。该策略专门针对新预测方式下的专家负载均衡问题进行了优化。

## 核心特性

### 1. 请求级别的负载追踪

- 引入 `RequestManager` 类管理请求状态
- 维护请求ID到索引的映射 (`m2`)
- 支持动态分配和回收请求索引

### 2. Token 到请求的映射

- 通过 `token2req` 张量建立 token 与请求的对应关系
- 支持批量 token 的请求索引分配
- 使用原地拷贝优化内存操作

### 3. 动态负载计算

- 区分 `moe_load_prev`（新策略）和 `moe_load`（旧策略）
- 支持 `pd_dynamic_decay` 参数控制动态衰减
- 公式：`all_moe_loads = all_moe_loads * pd_dynamic_decay + all_moe_loads_local * (1 - pd_dynamic_decay)`
- 结束请求后，`moe_load` 会移除已完成请求的 load 信息，避免陈旧负载数据影响后续调度

## 新增配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_batch_token` | int | 128 | 最大本地记录的 request 数 |
| `pd_dynamic_decay` | int | 0 | PD 动态衰减系数 |

## 使用方法

在 `ascend_config.json` 中配置：

```json
{
  "eplb_config": {
    "eplb_policy_type": 4,
    "max_batch_token": 128,
    "pd_dynamic_decay": 0
  }
}
```

## 策略类型说明

| 策略类型 | 说明 |
|----------|------|
| 0 | RandomLoadBalance |
| 1 | 默认策略 |
| 2 | SwiftEPLB |
| 3 | FlashLB |
| 4 | **PD Balancer（新增）** |

## 代码变更摘要

### 新增文件

- `vllm_ascend/eplb/core/policy/policy_pd_balancer.py` - PD Balancer 策略实现

### 修改文件

- `vllm_ascend/ascend_config.py` - 添加新配置参数
- `vllm_ascend/eplb/adaptor/vllm_adaptor.py` - 集成请求管理
- `vllm_ascend/eplb/core/policy/policy_factory.py` - 注册新策略
- `vllm_ascend/eplb/eplb_updator.py` - 更新负载计算逻辑
- `vllm_ascend/eplb/utils.py` - 添加负载获取和设置方法
- `vllm_ascend/ops/fused_moe/fused_moe.py` - MoE 层支持新策略
- `vllm_ascend/ops/fused_moe/moe_comm_method.py` - 传递 topk_ids
- `vllm_ascend/worker/model_runner_v1.py` - 集成请求槽位更新

## 提交信息

- **提交哈希**: 7057e6b0
- **提交时间**: 2026-04-27
- **提交类型**: [Feat]
- **提交描述**: Add a new eplb policy