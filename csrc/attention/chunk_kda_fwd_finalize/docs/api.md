# ChunkKdaFwdFinalize API

## Python

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_finalize

attn_out = chunk_kda_fwd_finalize(
    qg_scaled,
    aqk,
    v_new,
    h,
    output_layout="BSND",
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
)
```

这是 ctypes 直调 `aclnnChunkKdaFwdFinalize` 的稳定入口，不注册
legacy `torch.ops.npu` 接口。输入、输出和变长元数据的形状与约束
见下文及[算子 README](../README.md#输入输出)。

## aclnn

```cpp
aclnnStatus aclnnChunkKdaFwdFinalizeGetWorkspaceSize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwdFinalize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

四个 tensor 输入和 `attnOut` 必传，且 descriptor 必须为非私有
ND 格式；输入允许 L2 连续化，kernel 直接写的输出必须连续。
均为 BF16，`K=V=128`，`Aqk` 的列数为 64；支持 A2/A3/A5。
`outputLayout` 对 rank-4 `qgScaled/Aqk` 只接受 `BSND/BNSD`，
对 rank-3 `qgScaled/Aqk` 只接受 `TND/NTD`；packed 场景的
`vNew` 可为 FwdH 的 `[1,HV,T,128]` 或独立调用的
`[HV,T,128]`，区分大小写。输入形状和输出形状见
[算子 README](../README.md#输入输出)。

`stateVFirst=false` 时 `h` 表示 `[K,V]`；为 true 时表示 `[V,K]`。
`cuSeqlensOptional` 是严格递增、首 0 末 T 的 host int array。
有变长元数据时 rank-4 输入要求 `B=1`；`chunkIndicesOptional`
若提供必须与 `cuSeqlensOptional` 同时存在，且为规范的
sequence-major `(sequence_id,local_chunk_id)` 列表。

唯一输出 `attnOut` 是 BF16。`BSND/TND` 按 token 优先写出，
`BNSD/NTD` 按 value head 优先写出；所有输入仍然按 value head
优先排列。数值精度边界见[设计文档](design.md#数学与精度)。

## 返回码

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | workspace 查询和执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必传输入/输出或 workspaceSize/executor 为空 |
| `ACLNN_ERR_PARAM_INVALID` | dtype、shape、format、layout、连续性或变长元数据不合法 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | executor 创建失败 |
| `ACLNN_ERR_INNER_NULLPTR` | 中间 descriptor 创建失败 |
| `ACLNN_ERR_INNER` | kernel 调用失败 |
