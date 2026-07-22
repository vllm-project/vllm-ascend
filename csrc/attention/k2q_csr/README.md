# k2q_csr (KV Cache)

q2k → k2q CSR 自定义算子，接入 vllm-ascend `csrc/attention`。

## 调用

```python
from vllm_ascend.utils import enable_custom_op
enable_custom_op()

row_ptr, q_ind, slot = torch.ops._C_ascend.npu_k2q_csr(
    q2k,              # int32 [H, T, topk]
    cu_seqlens,       # int32 [B+1]
    cu_block_lens,    # int32 [B+1]
    0,                # order_method: 0=batch, 1=round-robin
    total_rows,       # >=0 推荐显式传入，避免 Host D2H
    max_kv,           # >=0 推荐显式传入
    1 if use_simt else 0,  # ascend950 Hist/Scatter SIMT
)
```

## 目录

- `k2q_csr_{meta,hist,row_prefix,tile_prefix,scatter}/`：五阶段 AscendC 算子
- `k2q_csr_common/`：共享 tiling / kernel 源
- `k2q_csr_torch_adpt.h`：`torch.ops._C_ascend.npu_k2q_csr` Host 编排

源码同步自 `xpu_kernel/C_like/transformer/npu/kvcache/k2q_csr`。
