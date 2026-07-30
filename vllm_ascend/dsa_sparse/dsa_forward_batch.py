"""DSA 单次 model-forward 的不可变批量数据契约。

本模块定义三类短生命周期只读视图：LIDU/KSC/SFA-Offload 共用的
``DSAForwardRowModeDecodeBatch``、逐层 lifecycle hook 共用的
``DSAForwardLayerHookPlan``，以及仅含对齐物理 src/dst 的
``DSAFullBlockDumpBatch``。这些对象描述“本轮算子消费什么”，不拥有请求账本、
resident/DRAM 资源，也不负责从 Python 请求对象创建 tensor。

语义来源是 ``DSAInputBatchState``；动态 eager 构造放在
``dsa_forward_batch_builder.py``；row-mode eager（含 request-major MTP）和
single-token graph 的固定地址镜像由 ``dsa_row_mode_runtime.py`` 管理。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DSALightningIndexerUpdateBuffers:
    """一个 attention layer 的 caller-owned LIDU 输出缓冲。

    四个 tensor 都由 worker-lifetime row-mode buffer owner 预分配。eager
    使用 active-row prefix，graph 使用 captured-row prefix；LIDU 通过 out
    ABI 原址写入，随后 KSC 和 SFA-Offload 在同一 layer 调用链内立即消费。
    """

    topk_index: torch.Tensor
    topk_slots: torch.Tensor
    miss_count: torch.Tensor
    tail_info: torch.Tensor


@dataclass(frozen=True)
class DSAFullBlockDumpBatch:
    """Layer-invariant HBM -> hot-DRAM full-block copy pairs.

    Request ownership, block hashes, logical block indices and DRAM allocation
    all belong to the model-forward control plane.  Once reservation finishes,
    every attention layer only needs these two aligned int32 tensors plus its
    own HBM cache and DRAM arena. Request rows are compacted into physical copy
    jobs before this view is exposed. Eager execution uses the real active-job
    prefix; graph execution uses a fixed captured-capacity view whose unused
    tail jobs carry destination ``-1``.
    """

    src_hbm_block_ids_tensor: torch.Tensor
    dst_dram_block_ids_tensor: torch.Tensor

    def __post_init__(self) -> None:
        src = self.src_hbm_block_ids_tensor
        dst = self.dst_dram_block_ids_tensor
        if not torch.is_tensor(src) or not torch.is_tensor(dst):
            raise TypeError("DSA full-block dump ids must be tensors")
        if src.ndim != 1 or dst.ndim != 1:
            raise ValueError("DSA full-block dump ids must be 1-D tensors")
        if int(src.numel()) != int(dst.numel()):
            raise ValueError("DSA full-block dump source/destination counts differ")

    @classmethod
    def empty(
        cls,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> DSAFullBlockDumpBatch:
        device = torch.device("cpu") if tensor_device is None else torch.device(tensor_device)
        return cls(
            src_hbm_block_ids_tensor=torch.empty((0,), dtype=torch.int32, device=device),
            dst_dram_block_ids_tensor=torch.empty((0,), dtype=torch.int32, device=device),
        )

    def __bool__(self) -> bool:
        return int(self.src_hbm_block_ids_tensor.numel()) > 0

    @property
    def row_count(self) -> int:
        return int(self.src_hbm_block_ids_tensor.numel())


@dataclass(frozen=True)
class DSAForwardRowModeDecodeBatch:
    """Model-forward-level row-mode decode metadata.

    Built once per model forward and reused by every attention layer. It
    contains the tensorized, layer-invariant inputs needed by the
    LIDU/KSC/SFA-Offload path: resident pool rows, row modes, HBM block tables,
    DRAM logical block tables, and caller-owned per-layer LIDU outputs.

    该视图覆盖完整 row-mode decode batch，其中 DENSE、SPARSE 和
    graph PAD 行由 ``row_modes_tensor`` 逐行区分。全 DENSE eager batch 通过
    ``uses_sparse_offload=False`` 保持原生 Indexer/SFA；只要存在 SPARSE 行，
    LIDU/KSC/SFA-Offload 就按完整 request-row batch 处理。

    当前 row-mode 合约是 decode-only：DSA 行与 request 行一一对齐。eager
    允许每个 request 行携带 1..N 个 request-major MTP token，attention
    按 round 建立 request-row -> flattened-token 映射；graph 仍只接受
    single-token 行。prefill/decode mixed batch 不在此合约内。
    """

    resident_pool_indices_tensor: torch.Tensor
    batch_hbm_block_table: torch.Tensor
    batch_dram_block_table: torch.Tensor
    # DENSE row 的 topK 逐层不同，因此 eager/graph 共用的 owner 为每层
    # 预留独立 LIDU 原址输出；forward view 只按当前 row prefix 切片。
    layer_lidu_outputs: tuple[DSALightningIndexerUpdateBuffers, ...] | None
    row_modes_tensor: torch.Tensor
    # Keep dense-only decode on the native Indexer/SFA path. This host scalar
    # is fixed while the forward view is built, so attention layers do not
    # inspect a device row-mode tensor or introduce an NPU->host sync.
    uses_sparse_offload: bool
    # Decode full-block dump remains an independent data-plane contract. Its
    # columns reuse the row-mode buffer owner, but they are not request-row
    # aligned: real physical copies are compacted into a job prefix and graph
    # padding carries dst=-1. Eager views only the real prefix; graph views the
    # fixed captured width.
    full_block_dump_batch: DSAFullBlockDumpBatch

    @classmethod
    def empty(
        cls,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> DSAForwardRowModeDecodeBatch:
        device = torch.device("cpu") if tensor_device is None else torch.device(
            tensor_device)
        empty_i32_table = torch.empty((0, 0), dtype=torch.int32, device=device)
        return cls(
            resident_pool_indices_tensor=torch.empty(
                (0,), dtype=torch.int32, device=device),
            batch_hbm_block_table=empty_i32_table,
            batch_dram_block_table=empty_i32_table,
            layer_lidu_outputs=None,
            row_modes_tensor=torch.empty((0,), dtype=torch.int32, device=device),
            uses_sparse_offload=False,
            full_block_dump_batch=DSAFullBlockDumpBatch.empty(tensor_device=device),
        )

    def __bool__(self) -> bool:
        return self.row_count > 0

    @property
    def row_count(self) -> int:
        return int(self.resident_pool_indices_tensor.numel())

    @property
    def max_logical_blocks(self) -> int:
        if self.batch_dram_block_table.ndim < 2:
            return 0
        return int(self.batch_dram_block_table.shape[1])

    def lidu_outputs_for_layer(
        self,
        layer_id: int,
    ) -> DSALightningIndexerUpdateBuffers:
        outputs = self.layer_lidu_outputs
        if outputs is None:
            raise RuntimeError(
                "DSA row-mode batch has no preallocated LIDU outputs")
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= len(outputs):
            raise IndexError(
                f"DSA LIDU layer {layer_id} is outside [0, {len(outputs)})")
        return outputs[layer_id]


@dataclass(frozen=True)
class DSAForwardLayerHookPlan:
    """Model-forward-level plan for layer lifecycle hooks.

    当前只向 ``attention_finished`` 暴露 layer-invariant 满块复制表。
    ``attention_begin`` 独立负责首次注册 cache zones，不读取本 plan。后续若
    新增其他逐层 lifecycle 动作，应继续在 model-forward 控制面一次规划，
    layer hook 只绑定 layer id 和 cache zones，不重新遍历请求。

    Dump completion is deliberately not represented here. The current DSA
    execution contract submits cache writes, full-block dump, and the next
    forward's LIDU/KSC consumption to the same NPU stream, so stream ordering
    is the dependency mechanism. If dump is moved to an independent stream
    later, it must use device events/completion state rather than a host
    boolean ledger.
    """

    full_block_dump_batch: DSAFullBlockDumpBatch

    @classmethod
    def empty(
        cls,
        *,
        tensor_device: torch.device | str | None = None,
    ) -> DSAForwardLayerHookPlan:
        return cls(
            full_block_dump_batch=DSAFullBlockDumpBatch.empty(tensor_device=tensor_device),
        )

    def __bool__(self) -> bool:
        return bool(self.full_block_dump_batch)
