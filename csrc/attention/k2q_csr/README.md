# k2q_csr (KV Cache)

q2k → k2q CSR 自定义算子，接入 vllm-ascend `csrc/attention`。

源码对齐 `xpu_kernel/C_like/transformer/npu/kvcache/k2q_csr`（含 MC FastB1 / 批量 MTE3 /
qTile ping-pong、`q_global_offset`、SIMT 路径隔离）。

## 调用

```python
from vllm_ascend.models.minimax_m3.ops.k2q_csr import npu_k2q_csr

row_ptr, q_ind, slot = npu_k2q_csr(
    q2k,              # int32 [H, T, topk]
    cu_seqlens,       # int32 [B+1]
    cu_block_lens,    # int32 [B+1]
    order_method=0,   # 0=batch/concat, 1=round-robin
    total_rows=total_rows,  # >=0 推荐显式传入，避免 Host D2H
    max_kv=max_kv,          # >=0 推荐显式传入
    use_simt=True,          # ascend950 Hist/Scatter SIMT；False=MC(A2/A5)
    q_global_offset=False,  # False=batch-local q_ind；True=全局 Q 下标
)
```

底层等价于 ``torch.ops._C_ascend.npu_k2q_csr(...)``（见
``vllm_ascend/models/minimax_m3/ops/k2q_csr.py``）。

| 参数 | 含义 |
|------|------|
| `use_simt=0` | `K2qCsrPipelineMc`（SIMD/MC，A2+A5） |
| `use_simt=1` | SIMT VF（仅 ascend950；tiling 在非 950 强制 0） |
| `q_global_offset=0` | `q_ind = qAbs - cu_q[bi]` |
| `q_global_offset=1` | `q_ind = qAbs` |

## 目录

- `k2q_csr_{meta,hist,row_prefix,tile_prefix,scatter}/`：五阶段 AscendC 算子
- `k2q_csr_common/`：共享 tiling / kernel 源
- `k2q_csr_torch_adpt.h`：`torch.ops._C_ascend.npu_k2q_csr` Host 编排

## 打包编译（含 kernel binary）

```bash
source /usr/local/Ascend/cann/set_env.sh   # 或本机 cann/ascend-toolkit 路径
cd /path/to/vllm-ascend/csrc

# A5（ascend950）；A2 用 --soc=ascend910b
bash build.sh --pkg --soc=ascend950 --ops=k2q_csr -j$(nproc)
bash build_out/cann-*-custom_linux-*.run --quiet
```

CMake 配置时会自动将 `k2q_csr_common/op_kernel` 同步到各阶段 `op_kernel/common/`（该目录
gitignore，勿手工依赖已 vendor 的副本）。

精度测试：

```bash
pytest tests/ut/ops/test_k2q_csr.py -v
```
