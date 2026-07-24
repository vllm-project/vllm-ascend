# FP8 Checkpoint 在 NPU 上的处理经验

## 背景

许多前沿模型（如 MiniMax M2.5）只发布 FP8 checkpoint。Ascend NPU 当前不支持原生 FP8 推理，需要在加载时反量化。

## 核心 Pattern

### 1. Config 层拦截

在 `patch_minimax_m2_config.py` 中拦截量化验证：

```python
def _patched_verify_quantization(self):
    if current_platform.device_name == "npu" and model_type == "minimax_m2" and quant_method == "fp8":
        cfg.quantization = None  # 关闭 FP8 量化路径，后续当 bf16 处理
```

### 2. Worker 层反量化

在 `patch_minimax_m2.py` 中包装 `load_weights`：

```python
def _patched_load_weights(self, weights):
    if self._need_dequantize_fp8_weights():
        weights = self._fp8_dequant_weight_iter(weights)  # fp8 → bf16
    return _original_load_weights(self, weights)
```

反量化公式：
```
bf16_weight = fp8_weight.to(bf16) * scale_inv.repeat_interleave(block_size)
```

### 3. 注意事项

- FP8 block size 从 `quantization_config.weight_block_size` 读取，常见为 `[128, 128]`
- weight 和 scale_inv 的加载顺序不保证，需要双向缓冲配对
- 反量化后直接丢弃 scale_inv，不存入 model state_dict

## 适用场景

此 Pattern 适用于所有「模型只发布 FP8 版本，需要在 NPU 上跑 bf16 推理」的情况。
