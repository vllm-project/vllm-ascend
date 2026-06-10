# 算子日志整改 开发说明文档

## PR标题

`[Misc][Ops] add triton ops log` — Triton 算子日志标准化整改

## 背景与目标

`vllm_ascend` 的 Triton 算子此前缺乏统一的日志输出，错误路径直接使用裸 `assert`/`raise`，导致问题难以定位。本次整改目标：

- 统一所有 Triton 算子的日志格式，以 `[TritonOps]` 为前缀
- 在算子入口处以 `debug` 级别打印输入 shape，便于问题复现
- 将裸 `assert`/`raise` 错误路径改为先 `logger.error()` 再 `raise`，确保错误进入结构化日志
- 修复 `causal_conv1d` 中硬编码 `CORE_HINT` 的问题

---

## 变更范围

共涉及 16 个文件，全部位于 `vllm_ascend/ops/triton/` 目录下。

---

## 核心变更模式

### 模式一：算子入口添加 debug 日志（打印输入 shape）

**适用文件：** `rope.py`, `muls_add.py`, `mamba/causal_conv1d.py`, `layernorm_gated.py`, `linearnorm/split_qkv_rmsnorm_rope.py`, `linearnorm/split_qkv_rmsnorm_mrope.py` 等所有公开算子函数

**规范格式：**
```python
logger.debug(
    f"[TritonOps] <函数名>: x.shape={x.shape}, weight.shape={weight.shape}, ...")
```

**示例** (`rope.py:rope_forward_triton`)：
```python
logger.debug(
    f"[TritonOps] rope_forward_triton: q.shape={q.shape}, k.shape={k.shape}, "
    f"rope_dim={rope_dim}, is_neox_style={is_neox_style}"
)
```

---

### 模式二：裸 `assert`/`raise RuntimeError` 替换为 `logger.error()` + `raise`

**适用文件：** `triton_utils.py`, `layernorm_gated.py`, `mamba/causal_conv1d.py`

**变更前：**
```python
assert HAS_TRITON, "..."
raise RuntimeError("Feature dim too large.")
```

**变更后：**
```python
if not HAS_TRITON:
    logger.error("[TritonOps] Failed to resolve Triton op '%s' because HAS_TRITON is False.", op_name)
    raise RuntimeError("[TritonOps] ...")
```

**`layernorm_gated.py:layer_norm_fwd_npu`** 示例：
```python
if group_size > BLOCK_N:
    raise RuntimeError(
        f"Feature dim too large: group_size={group_size} "
        f"exceeds BLOCK_N={BLOCK_N} "
        f"(MAX_FUSED_SIZE={MAX_FUSED_SIZE})")
```
（错误信息增加了具体数值，便于诊断）

---

### 模式三：assert 错误信息增强

**适用文件：** `muls_add.py`

**变更前：**
```python
assert x.shape == y.shape
```

**变更后：**
```python
assert x.shape == y.shape, (
    f"Input tensors must have the same shape, "
    f"got x.shape={x.shape} and y.shape={y.shape}")
```

---

### 模式四：ImportError 静默失败改为 warning 日志

**适用文件：** `triton_utils.py`

**变更前：**
```python
try:
    import triton.language.extra.cann.extension as _extension_module
except ImportError:
    _extension_module = None
```

**变更后：**
```python
except ImportError:
    logger.warning(
        "[TritonOps] Failed to import "
        "triton.language.extra.cann.extension, "
        "falling back to triton.language for op resolution."
    )
    _extension_module = None
```

---

### 模式五：Triton 自定义 op 解析成功后添加 debug 日志

**适用文件：** `triton_utils.py`

```python
insert_slice = _resolve_triton_ascend_op("insert_slice")
extract_slice = _resolve_triton_ascend_op("extract_slice")
get_element = _resolve_triton_ascend_op("get_element")
logger.debug("[TritonOps] Resolved triton ascend ops: insert_slice, extract_slice, get_element")
```

---

## `triton_utils.py` 整体结构说明

`triton_utils.py` 是所有 Triton 算子的基础工具模块，本次整改后其职责如下：

| 函数/变量 | 说明 |
|---|---|
| `_extension_module` | CANN Triton 扩展模块，导入失败时 warning 并回退 |
| `_resolve_triton_ascend_op(op_name)` | 解析 Ascend 自定义 Triton op，失败时 `logger.error` + `raise` |
| `insert_slice / extract_slice / get_element` | 模块加载时解析，成功后打印 debug 日志 |
| `init_device_properties_triton()` | 初始化 NPU 设备属性（aicore 数、vectorcore 数），失败时 `logger.error` + `raise` |
| `get_aicore_num()` / `get_vectorcore_num()` | 获取核心数，未初始化时 `logger.error` + `raise` |

---

## 日志级别规范

| 场景 | 级别 |
|---|---|
| 算子入口 + 输入 shape | `logger.debug` |
| op 解析/初始化成功 | `logger.debug` |
| 可选依赖导入失败（有回退） | `logger.warning` / `logger.warning_once` |
| 不可恢复错误（raise 前） | `logger.error` |

---

## 测试方式

### 验证日志输出

设置日志级别为 `DEBUG`，运行任意 Triton 算子测试，观察 `[TritonOps]` 前缀日志是否出现：

```bash
# 设置 vllm 日志级别
export VLLM_LOGGING_LEVEL=DEBUG

# 运行单个算子测试（需要 NPU 环境）
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_causal_conv1d.py -v
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py -v
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope.py -v
```

预期 debug 日志输出示例：
```
DEBUG vllm: [TritonOps] causal_conv1d_update_npu: x.shape=torch.Size([2, 64]), conv_state.shape=...
DEBUG vllm: [TritonOps] rope_forward_triton: q.shape=torch.Size([128, 32, 128]), k.shape=...
DEBUG vllm: [TritonOps] Resolved triton ascend ops: insert_slice, extract_slice, get_element
```

### 验证错误路径日志

构造非法输入触发错误，确认 `logger.error` 先于异常输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 触发 shape 不匹配错误（muls_add）
import torch
from vllm_ascend.ops.triton.muls_add import muls_add_triton
x = torch.randn(4, dtype=torch.float16)
y = torch.randn(8, dtype=torch.float16)
muls_add_triton(x, y, 1.0)  # 应先打印 ERROR 日志再抛 AssertionError

# 触发 activation 非法错误（causal_conv1d）
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_ref
causal_conv1d_ref(x.unsqueeze(0).unsqueeze(0),
                  torch.randn(1, 4),
                  activation="relu")  # 应先打印 ERROR 日志再抛 NotImplementedError
```

---

## 注意事项

1. **无用户可见行为变化**：所有改动仅影响日志输出，不改变算子的计算逻辑和输出结果。
2. **debug 日志默认不输出**：`logger.debug` 仅在日志级别设为 `DEBUG` 时生效，线上环境无性能影响。
3. **`get_vectorcore_num()` 依赖初始化**：调用前须确保 `init_device_properties_triton()` 已执行，否则会抛出带有明确说明的 `RuntimeError`。
4. **`[TritonOps]` 前缀**：所有日志消息统一添加此前缀，便于在混合日志中过滤 Triton 算子相关输出。
