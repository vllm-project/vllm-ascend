# Top-K 采样参数测试套件

## 概述

本测试套件针对 vLLM 推理引擎的 `top_k` 采样参数进行全面测试，覆盖正常场景、异常场景、边界值以及效果验证。

## 测试文件说明

| 文件名 | 测试类型 | 用例数 | 描述 |
|--------|----------|--------|------|
| `test_top_k_normal.py` | 正常场景 | 4个 | 常规取值范围内(top_k=1,10,50等)的测试 |
| `test_top_k_abnormal.py` | 异常场景 | 12个 | 非法值、类型错误、null等异常场景 |
| `test_top_k_boundary.py` | 边界值 | 6个 | 边界值验证(0, 1, -1, -2, 超大值等) |
| `test_top_k_effect.py` | 效果验证 | 4个 | 验证top_k对生成结果多样性和质量的影响 |

## 测试覆盖范围

### 正常场景 (test_top_k_normal.py)

- ✅ `test_top_k_normal_values`: top_k正常取值[1, 10, 50, 100, 1000]
- ✅ `test_top_k_with_temperature`: top_k与temperature组合
- ✅ `test_top_k_with_top_p`: top_k与top_p组合(核采样)
- ✅ `test_top_k_disable_with_minus_one`: top_k=-1禁用限制

### 异常场景 (test_top_k_abnormal.py)

- ❌ `test_top_k_zero_non_stream`: top_k=0 非流式场景
- ❌ `test_top_k_zero_stream`: top_k=0 流式场景
- ❌ `test_top_k_negative_not_minus_one_non_stream`: top_k<-1 非流式场景
- ❌ `test_top_k_negative_not_minus_one_stream`: top_k<-1 流式场景
- ❌ `test_top_k_float_non_stream`: top_k为浮点数 非流式
- ❌ `test_top_k_float_stream`: top_k为浮点数 流式
- ❌ `test_top_k_string_non_stream`: top_k为字符串 非流式
- ❌ `test_top_k_string_stream`: top_k为字符串 流式
- ❌ `test_top_k_null_non_stream`: top_k为null 非流式
- ❌ `test_top_k_null_stream`: top_k为null 流式
- ⚠️ `test_top_k_exceed_vocab_size_non_stream`: top_k超过词表大小 非流式
- ⚠️ `test_top_k_exceed_vocab_size_stream`: top_k超过词表大小 流式

### 边界值 (test_top_k_boundary.py)

- 🔹 `test_top_k_boundary_1`: 边界值=1(接近贪婪解码)
- 🔹 `test_top_k_very_large`: 极大值=100000
- 🔹 `test_top_k_minus_one_boundary`: -1标记边界
- 🔹 `test_top_k_without_setting`: 未设置参数(使用默认值)
- 🔹 `test_top_k_zero_boundary_non_stream`: 0值边界
- 🔹 `test_top_k_minus_two_boundary`: -2值边界

### 效果验证 (test_top_k_effect.py)

- 🎯 `test_top_k_small_conservative`: 小top_k(=5)保守采样
- 🎯 `test_top_k_large_diverse`: 大top_k(=100)多样性
- 🎯 `test_top_k_disabled_full_vocab`: top_k=-1禁用限制的完整词表采样
- 🎯 `test_top_k_combined_with_top_p_priority`: 与top_p的协同工作

## 关键验证点

1. **HTTP状态码**: 验证200(正常)或400(异常)
2. **错误码**: 验证返回的 error code 是否符合预期
3. **finish_reason**: 验证stop/length等结束原因
4. **流式[DONE]**: 流式响应是否包含[DONE]标记
5. **结果有效性**: 验证生成内容不为空且格式正确

## 执行方式

```bash
# 执行所有top_k测试
pytest tests/top_k/ -v

# 仅执行正常场景
pytest tests/top_k/test_top_k_normal.py -v

# 仅执行异常场景
pytest tests/top_k/test_top_k_abnormal.py -v

# 仅执行边界值测试
pytest tests/top_k/test_top_k_boundary.py -v

# 仅执行效果验证
pytest tests/top_k/test_top_k_effect.py -v

# 执行特定测试
pytest tests/top_k/test_top_k_normal.py::test_top_k_normal_values -v
```

## 注意事项

1. 异常场景中，**流式请求**返回状态码200但响应体包含错误码400；**非流式请求**直接返回状态码400
2. top_k超过词表大小时，系统可能自动钳制到词表大小而非报错
3. top_k=0被视为无效值，需要明确返回错误
4. 只有top_k=-1表示禁用，其他负数均为非法

## 参数说明

| 参数值 | 含义 | 预期行为 |
|--------|------|----------|
| 1 ~ vocab_size | 限制候选token数量 | 正常采样 |
| -1 | 禁用top_k限制 | 使用全部词表 |
| 0 | 无效值 | 返回400错误 |
| < -1 | 无效值 | 返回400错误 |
| 浮点数 | 类型错误 | 返回400错误 |
| 字符串 | 类型错误 | 返回400错误 |
| null | 无效值 | 返回400错误 |

## 参考文档

- vLLM SamplingParams: https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- OpenAI API: https://platform.openai.com/docs/api-reference/completions
