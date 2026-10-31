# LightningIndexerV2 bindings

The bindings expose the CANN `aclnnLightningIndexerV2` and
`aclnnLightningIndexerV2Metadata` interfaces through the existing vllm-ascend
extension. They do not bundle operator kernels or change the model QSA path.

## Requirements

Build the vllm-ascend C++ extension and install a compatible CANN transformer
operator library on supported A2, A3 or 950 hardware. Both execution entry
points and their `GetWorkspaceSize` entry points must be discoverable by the
existing ACLNN loader. Installing a `cann_ops_transformer` Python package is
not required. Hardware-specific restrictions are checked by CANN.

## Usage

```python
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C
from vllm_ascend.ops.lightning_indexer_v2 import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
)

# q: [T,H,128], k: [pages,page_size,1,128], w: [T,H] FP32.
# cu_q: INT32 cumulative query lengths including the initial zero.
# used_k: INT32 compressed-key lengths per request; table: INT32 page table.
common = dict(
    cu_seqlens_q=cu_q,
    seqused_k=used_k,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=0,
    cmp_ratio=1,
)
metadata = lightning_indexer_v2_metadata(
    q.shape[1], 1, 128, 512, batch_size=used_k.shape[0], **common
)
indices, values = lightning_indexer_v2(
    q, k, w, 512, metadata=metadata, block_table=table,
    return_value=0, **common
)
```

Metadata is a 2048-element INT32 NPU tensor. At least one NPU length tensor
is required for device dispatch; all-None tensor inputs are not supported.
Optional metadata scalar arguments accept `None` with
upstream defaults. All provided tensors must belong to the intended device.

Indices are INT32 with shape `[T,Nk,topk]` for TND queries or
`[B,Sq,Nk,topk]` for BSND queries. Values are FP32, with the same shape when
`return_value=1`, otherwise shape `[0]`. Invalid entries are `-1` and `-inf`.
The paged key cache may have a noncontiguous page stride; the wrapper does
not copy it. The interface only exposes ordinary V2 selection, not V3 candidate
outputs. `cmp_ratio` configures masking; it does not compress input keys.

## Validation

Run the operator tests on a device after building the extension:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_lightning_indexer_v2.py
```

Coverage includes paged cache strides, compressed causal visibility and residuals,
short candidate lists, values, TND/BSND layouts, Meta dispatch,
compilation, and graph replay with changed input weights. These tests require
both CANN operators; missing operator symbols are failures rather than skips.

The bridge preserves the upstream ACLNN argument order, metadata empty-tensor
normalization and output protocol. It uses the project's `EXEC_NPU_CMD` rather
than upstream `ACLNN_CMD`; upstream stream-resource and deterministic-level
configuration are not newly introduced here. Device guards are set in the new
wrappers. Hardware validation must cover the deployed torch-npu/CANN versions,
including any required deterministic-level behavior.
