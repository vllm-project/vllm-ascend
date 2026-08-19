# MsaIndexScore 测试说明

## 1. 端到端精度自验证（当前主用例）

`examples/test_aclnn_msa_index_score.cpp` 自包含 aclnn 调用 + CPU golden。

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

## 2. 用例矩阵

对齐设计文档黄金用例；`start_loc` 为**逻辑 block 索引**，因果由 `sparse_mode=3` 承担。

| 用例 | 场景 | 覆盖点 |
|------|------|--------|
| `L0-debug-trace` | 极小尺寸 | 主路径 / TRACE |
| `L0-int8-dequant-trace` | int8 + scale | 前融合反量化 |
| `L0-prefill-aligned` | chunked prefill 对齐 | rightDownCausal + local_mask |
| `L1-prefill-unaligned` | 多 batch varlen | 边界 block mask |
| `L1-prefill-multi-mtile` | 行数 > M-tile | M-tile 切分 |
| `L1-decode-lq1` | decode q_len=1 | 多长度 |
| `L1-decode-speculative` | q_len>1 | 投机解码 |
| `L1-long-seq-multi-stile` | kv=4096 | 多 S-tile |
| `L1-bf16` / `L1-int8-dequant` | dtype | 非量化 / 量化 |
| `L2-tiny-kv` | 极小 kv | 尾填充 |
| `L1-bnbd` / `L1-bnbd-int8` | PA BNBD | `[NP, N2, P, D]` |
| `L1-tnd-unaligned` / `L1-tnd-int8` / `L0-tnd-tiny` | TND packed | 无 block_table，klen 前缀和 |

默认跑完整用例矩阵（含 TND / BNBD）。key 布局由 `layout_key`（aclnn：`layoutKeyOptional`）指定，不再从 shape 推断。

## 3. Python 参考

`tests/golden/msa_index_score_golden.py`：

```python
golden = msa_index_score_golden(
    query, key, block_table, actual_seq_qlen, actual_seq_klen, start_loc,
    sparse_mode=3, scale=None)
```

## 4. 判定标准

- 填充位（不可见 block）两侧同为 `-inf`
- `local_mask` 强制高分两侧同为 `≥1e28`
- 有效位 `atol/rtol=1e-3`，`error_ratio≤1e-3`
