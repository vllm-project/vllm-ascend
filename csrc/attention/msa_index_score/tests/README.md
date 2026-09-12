# MsaIndexScore 测试说明

## 1. 端到端精度自验证（当前主用例）

`examples/test_aclnn_msa_index_score.cpp` 自包含 aclnn 调用 + CPU golden。

```bash
# Atlas A2/A3
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom --soc=ascend910b
# 通过：末行 [PASS]: 36/36 cases passed（A2 跳过 4 条 FP8）

# Ascend 950：必须 --soc=ascend950；安装后 source vendors/custom_transformer/bin/set_env.bash
bash build.sh --pkg --soc=ascend950 --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=/tmp/msa_opp
source /tmp/msa_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom --soc=ascend950
# 通过：末行 [PASS]: 40/40 cases passed
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
| `L0-fp8-e4m3fn` / `L0-fp8-e5m2` / `L1-fp8-e4m3fn-prefill` | 950 FP8（D=128） | Cube 原生 e4m3fn / e5m2；hifloat8 仅 kernel |
| `L1-pad-q0` | 部分请求 `q_len=0` | mixed-batch 跳过空 query |
| `L1-pad-kv0` | 部分请求 `kv_len=0` | mixed-batch 空 KV，score 填 `-inf` |
| `L1-pad-q0-kv0` / `L1-tnd-pad-q0-kv0` | 头部 `q_len=kv_len=0` | PA / TND 同时 pad |
| `L1-pad-mid-q0` | 中间请求 `q_len=kv_len=0` | 前后有效请求夹空 pad |
| `L0-all-q0` / `L1-all-q0` | 整 batch `q_len=0` | host 不拦截，跳过计算 |
| `L0-all-kv0` | 整 batch `kv_len=0` | 跳过 QK，score 全 `-inf` |
| `L0-all-q0-kv0` | 整 batch 两侧 0 | 空 query + 空 KV |
| `L0-tnd-all-q0` / `L0-tnd-all-kv0` / `L0-tnd-all-q0-kv0` | TND 整 batch 空 | packed key / 空 key 张量 |
| `L0-stride-bbnd` / `L1-stride-bbnd` | PA BBND dim0 gap=2 | 间隔槽下毒，校验 stride 寻址 |
| `L1-stride-bnbd` | PA BNBD dim0 gap=2 | 另一 PA 布局 |
| `L1-stride-int8` | PA int8 dim0 gap=2 | 量化拷页 + stride |
| `L0-wide-table-257` / `L1-wide-table-257-bf16` | PA `block_table` 宽 257 | 950 C2UB 256 列滑窗 flush |
| `L0-fp8-wide-table-257` | 同上 + FP8 | score 末维 272，有效列与 fill 位 |

默认跑完整用例矩阵（含 TND / BNBD、mixed-batch pad、整 batch `q_len`/`kv_len=0`、PA key dim0 stride、宽 `block_table`）。950 另含 4 条 FP8，共 40 条；A2/A3 跳过 FP8，期望 36/36。key 布局由 `layout_key`（aclnn：`layoutKeyOptional`）指定，不再从 shape 推断。

## 3. Python 参考

`tests/golden/msa_index_score_golden.py`：

```python
golden = msa_index_score_golden(MsaIndexScoreGoldenInputs(
    query, key, block_table, actual_seq_qlen, actual_seq_klen, start_loc,
    sparse_mode=3, scale=None))
```

## 4. 判定标准

- 填充位（不可见 block）两侧同为 `-inf`
- `local_mask` 强制高分两侧同为 `≥1e28`
- 有效位 `atol/rtol=1e-3`，`error_ratio≤1e-3`；950 FP8 为 `2e-2`
