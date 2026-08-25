import torch
import numpy as np
import ctypes
import random

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

# Keep in sync with sparse_attention_score_metadata.h.
METADATA_TOTAL_SIZE = 1024
METADATA_MAGIC = 0x5341534D
METADATA_VERSION = 6
METADATA_USED_SIZE = 272
SA_USED_CORE_NUM_INDEX = 3
SA_TOTAL_TASK_NUM_INDEX = 4
DEFAULT_AIC_CORE_NUM = 24


def safe_to_tensor(arr):
    if arr is None:
        return None
    if isinstance(arr, list):
        arr = np.array(arr)
    if isinstance(arr, torch.Tensor):
        return arr
    if hasattr(arr, "dtype") and arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


def calc_sa_total_task_num(q_seqlens, num_q_heads, num_kv_heads, is_packed_gqa=1):
    """Match AICPU BuildSaTaskInfo: totalQTokenNum * sparseHeadNum."""
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(f"invalid head nums: Nq={num_q_heads}, Nkv={num_kv_heads}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Nq={num_q_heads} must be divisible by Nkv={num_kv_heads}")
    total_q_tokens = int(sum(int(s) for s in q_seqlens))
    sparse_head_num = int(num_kv_heads) if int(is_packed_gqa) == 1 else int(num_q_heads)
    return total_q_tokens * sparse_head_num


def simulate_aicpu_metadata(
    q_seqlens,
    num_q_heads,
    num_kv_heads,
    is_packed_gqa=1,
    aic_core_num=DEFAULT_AIC_CORE_NUM,
    device=None,
):
    """Host stand-in for AICPU EncodeMetadata: INT32[1024] header."""
    if aic_core_num <= 0:
        raise ValueError(f"aic_core_num must be > 0, got {aic_core_num}")
    sa_total_task_num = calc_sa_total_task_num(
        q_seqlens, num_q_heads, num_kv_heads, is_packed_gqa=is_packed_gqa
    )
    metadata = torch.zeros(METADATA_TOTAL_SIZE, dtype=torch.int32, device=device)
    metadata[0] = METADATA_MAGIC
    metadata[1] = METADATA_VERSION
    metadata[2] = METADATA_USED_SIZE
    metadata[SA_USED_CORE_NUM_INDEX] = min(sa_total_task_num, int(aic_core_num))
    metadata[SA_TOTAL_TASK_NUM_INDEX] = sa_total_task_num
    return metadata


def prefix_sum_from_seqlens(seqlens):
    out = [0]
    for s in seqlens:
        out.append(out[-1] + int(s))
    return out


def _optional_int_vec(t, batch):
    if t is None:
        return None
    t = safe_to_tensor(t)
    if t is None:
        return None
    t = t.cpu().to(torch.int64).view(-1)
    if t.numel() != batch:
        return None
    return [int(x) for x in t.tolist()]


def decode_cu_and_seqused(cu_q, cu_kv, seqused_q=None, seqused_kv=None, total_q=None):
    """Decode storage (cu) and actual (seqused if present) per-batch lengths."""
    if cu_q is None or cu_kv is None:
        return None
    cu_q = safe_to_tensor(cu_q).cpu().to(torch.int64).view(-1)
    cu_kv = safe_to_tensor(cu_kv).cpu().to(torch.int64).view(-1)
    if cu_q.numel() < 2 or cu_kv.numel() != cu_q.numel():
        return None
    if int(cu_q[0].item()) != 0 or int(cu_kv[0].item()) != 0:
        return None
    batch = cu_q.numel() - 1
    storage_q = [(cu_q[i + 1] - cu_q[i]).item() for i in range(batch)]
    storage_kv = [(cu_kv[i + 1] - cu_kv[i]).item() for i in range(batch)]
    if any(s < 0 for s in storage_q) or any(s < 0 for s in storage_kv):
        return None
    if total_q is not None and sum(storage_q) != int(total_q):
        return None
    used_q = _optional_int_vec(seqused_q, batch)
    used_kv = _optional_int_vec(seqused_kv, batch)
    actual_q = used_q if used_q is not None else storage_q
    actual_kv = used_kv if used_kv is not None else storage_kv
    if any(a < 0 or a > s for a, s in zip(actual_q, storage_q)):
        return None
    if any(a < 0 or a > s for a, s in zip(actual_kv, storage_kv)):
        return None
    # actual==0 allowed (padding / empty request); only reject when both >0 and kv < q.
    if any(q > 0 and kv > 0 and kv < q for q, kv in zip(actual_q, actual_kv)):
        return None
    return actual_q, actual_kv, storage_q, storage_kv


def seqlens_from_cu_seq(cu_q, cu_kv, total_q=None, seqused_q=None, seqused_kv=None):
    """Actual seqlens (seqused if present, else cu). None if invalid."""
    decoded = decode_cu_and_seqused(
        cu_q, cu_kv, seqused_q=seqused_q, seqused_kv=seqused_kv, total_q=total_q
    )
    if decoded is None:
        return None
    return decoded[0], decoded[1]


def recover_batch_seqlens(query, sparse_block_idx, block_table, metadata=None):
    """Equal-batch seqlens: q=T/B, kv=maxBlocks*128 (metadata is INT32[1024], not kv len)."""
    del sparse_block_idx, metadata
    batch = int(block_table.shape[0])
    total_q = int(query.shape[0])
    max_blocks = int(block_table.shape[1])
    if batch <= 0 or total_q % batch != 0:
        raise ValueError(
            f"Cannot recover equal-batch q seqlens: T={total_q}, B={batch}"
        )
    q_seqlen = total_q // batch
    kv_seqlen = max(max_blocks * 128, q_seqlen)
    return [q_seqlen] * batch, [kv_seqlen] * batch


def generate_block_table(batch, max_blocks_per_batch, num_physical, seed=10, identity=False):
    """logical->physical map. Smoke cases use identity; otherwise shuffle like test_bf16."""
    need = batch * max_blocks_per_batch
    if need > num_physical:
        raise ValueError(f"need {need} physical ids but key only has {num_physical}")
    block_table = torch.full((batch, max_blocks_per_batch), -1, dtype=torch.int32)
    if identity or need == num_physical == 1:
        for b in range(batch):
            for i in range(max_blocks_per_batch):
                block_table[b, i] = b * max_blocks_per_batch + i
        return block_table
    rng = random.Random(seed)
    all_physical = list(range(num_physical))
    rng.shuffle(all_physical)
    chosen = all_physical[:need]
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = chosen[b * max_blocks_per_batch + i]
    return block_table


def build_sparse_and_block_table(
    q_seqlens,
    kv_seqlens,
    kv_heads,
    top_k,
    num_physical,
    block_shape_y=128,
    seed=10,
    q_storage_seqlens=None,
):
    """Build TND packed sparseBlockIdx/Count [Nkv, totalQBlocks, ...] and blockTable.

    totalQBlocks follows storage (cu). Only actual q tokens (seqused) get filled rows.
    Smoke (max_blocks==1, topK==1): always select logical block 0, identity blockTable.
    """
    np.random.seed(seed)
    random.seed(seed)
    batch = len(q_seqlens)
    if q_storage_seqlens is None:
        q_storage_seqlens = q_seqlens
    total_q_blocks = sum(int(s) for s in q_storage_seqlens)
    max_blocks = max((kv + block_shape_y - 1) // block_shape_y for kv in kv_seqlens)
    top_k = min(top_k, max_blocks)
    smoke = (max_blocks == 1 and top_k == 1)

    sparse_idx = torch.full((kv_heads, total_q_blocks, top_k), -1, dtype=torch.int32)
    sparse_count = torch.zeros((kv_heads, total_q_blocks), dtype=torch.int32)
    block_table = generate_block_table(
        batch, max_blocks, num_physical, seed=seed, identity=smoke
    )

    q_storage = 0
    for b in range(batch):
        if int(q_seqlens[b]) == 0 or int(kv_seqlens[b]) == 0:
            q_storage += int(q_storage_seqlens[b])
            continue
        history_len = kv_seqlens[b] - q_seqlens[b]
        for q_block in range(q_seqlens[b]):
            global_q = q_storage + q_block
            last_logical = (history_len + q_block) // block_shape_y
            candidates = list(range(last_logical + 1))
            if not candidates:
                continue
            for h in range(kv_heads):
                if smoke:
                    selected = [0]
                else:
                    must = last_logical
                    others = [c for c in candidates if c != must]
                    num_select = random.randint(1, min(len(candidates), top_k))
                    selected = [must]
                    remain = num_select - 1
                    if remain > 0 and others:
                        selected.extend(random.sample(others, min(remain, len(others))))
                    selected = sorted(set(selected))[:top_k]
                sparse_count[h, global_q] = len(selected)
                for j, lid in enumerate(selected):
                    sparse_idx[h, global_q, j] = lid
        q_storage += int(q_storage_seqlens[b])

    return sparse_idx, sparse_count, block_table


def apply_init_tensors(input_data, device="cpu"):
    """Rewrite cuSeq / sparse / blockTable to legal values; keep QKV as generated."""
    query = safe_to_tensor(input_data.kwargs["query"])
    key = safe_to_tensor(input_data.kwargs["key"])
    sparse_block_idx = safe_to_tensor(input_data.kwargs["sparseBlockIdx"])
    block_table = safe_to_tensor(input_data.kwargs["blockTableOptional"])
    storage_q, storage_kv = recover_batch_seqlens(
        query,
        sparse_block_idx,
        block_table,
    )
    kv_heads = int(sparse_block_idx.shape[0])
    top_k = int(sparse_block_idx.shape[2])
    num_physical = int(key.shape[0])
    # Vary sparse pattern by case shape (still deterministic).
    seed = (
        10
        + int(query.shape[0]) * 17
        + int(query.shape[1]) * 31
        + num_physical * 13
        + top_k * 7
        + int(block_table.shape[0]) * 3
        + int(block_table.shape[1]) * 5
    )
    # Sparse seqused pad: only ~1/10 cases (need all batch q_seqlen>=2).
    use_seqused = all(int(s) >= 2 for s in storage_q) and (seed % 10 == 0)
    actual_q = (
        [int(s) - 1 for s in storage_q] if use_seqused else [int(s) for s in storage_q]
    )
    actual_kv = [int(s) for s in storage_kv]

    sparse_idx, sparse_count, block_table_new = build_sparse_and_block_table(
        actual_q,
        actual_kv,
        kv_heads,
        top_k,
        num_physical=num_physical,
        block_shape_y=128,
        seed=seed,
        q_storage_seqlens=storage_q,
    )
    cu_q = torch.tensor(prefix_sum_from_seqlens(storage_q), dtype=torch.int64)
    cu_kv = torch.tensor(prefix_sum_from_seqlens(storage_kv), dtype=torch.int64)
    seqused_q = torch.tensor(actual_q, dtype=torch.int32) if use_seqused else None
    seqused_kv = torch.tensor(actual_kv, dtype=torch.int32) if use_seqused else None

    num_heads = int(query.shape[1])
    is_packed_gqa = int(input_data.kwargs.get("isPackedGQA", 1))
    metadata = simulate_aicpu_metadata(
        actual_q,
        num_heads,
        kv_heads,
        is_packed_gqa=is_packed_gqa,
    )
    if metadata.numel() != METADATA_TOTAL_SIZE:
        raise RuntimeError(f"simulated metadata size must be {METADATA_TOTAL_SIZE}")

    if device in ("pyaclnn", "npu"):
        sparse_idx = sparse_idx.npu()
        sparse_count = sparse_count.npu()
        block_table_new = block_table_new.npu()
        cu_q = cu_q.npu()
        cu_kv = cu_kv.npu()
        if use_seqused:
            seqused_q = seqused_q.npu()
            seqused_kv = seqused_kv.npu()
        metadata = metadata.npu()

    input_data.kwargs["sparseBlockIdx"] = sparse_idx
    input_data.kwargs["sparseBlockCount"] = sparse_count
    input_data.kwargs["blockTableOptional"] = block_table_new
    input_data.kwargs["cuSeqLengthsQOptional"] = cu_q
    input_data.kwargs["cuSeqLengthsKvOptional"] = cu_kv
    input_data.kwargs["sequsedQOptional"] = seqused_q
    input_data.kwargs["sequsedKvOptional"] = seqused_kv
    input_data.kwargs["metadataOptional"] = metadata
    input_data.kwargs["winLeft"] = -1
    input_data.kwargs["winRight"] = -1
    input_data.kwargs["dstTypeMax"] = 0.0
    # softmaxPrecision legality is owned by tests/{st,950} generators + host tiling:
    #   st(910B): bf16→0, fp16→0|1;  950(A5): always 1.

    return actual_q, actual_kv


@register("aclnn_genericblocksparseattentioninputprocess")
class GenericBlockSparseAttentionInputProcess(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(GenericBlockSparseAttentionInputProcess, self).__init__(task_result, backend)
        self.return_lse = 0

    def init_by_input_data(self, input_data):
        np.random.seed(10)
        torch.npu.synchronize()
        apply_init_tensors(input_data, device="npu")

        input_args = []
        name_to_idx = {}
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            start = len(input_args)
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
            # Most ATK converts yield one ctypes object per named input.
            if data:
                name_to_idx[name] = start

        output_packages = []
        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

        # Null unused optionals by name (do NOT touch blockTableOptional / metadata).
        for name in (
            "attenMaskOptional",
            "qDequantScaleOptional",
            "kDequantScaleOptional",
            "vDequantScaleOptional",
            "pQuantScaleOptional",
        ):
            if name in name_to_idx:
                input_args[name_to_idx[name]] = null_tensor_ptr
        for name in ("sequsedQOptional", "sequsedKvOptional"):
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_tensor_ptr

        # Force required / used device tensors onto NPU, including blockTable + metadata.
        force_npu = [
            "query",
            "key",
            "value",
            "sparseBlockIdx",
            "sparseBlockCount",
            "metadataOptional",
            "cuSeqLengthsQOptional",
            "cuSeqLengthsKvOptional",
            "blockTableOptional",
        ]
        if input_data.kwargs.get("sequsedQOptional") is not None:
            force_npu.extend(["sequsedQOptional", "sequsedKvOptional"])
        for name in force_npu:
            if name not in name_to_idx:
                raise RuntimeError(f"missing required ATK input after convert: {name}")
            idx = name_to_idx[name]
            tensor = self.acl_tensor_to_torch(input_args[idx]).to("npu")
            input_args[idx] = self.torch_tensor_to_acl(tensor)

        self.return_lse = int(input_data.kwargs.get("returnSoftmaxlse", 0))
        output_packages = []
        if self.return_lse == 1:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float32)
            output.append(temp)
        if len(output) == 2:
            return output[0], output[1]
        return output[0]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, "
            "const aclTensor *metadataOptional, const aclTensor *attenMaskOptional, "
            "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
            "const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional, "
            "const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, "
            "const aclTensor *blockTableOptional, const aclIntArray *blockShape, "
            "int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue, "
            "int64_t maskType, int64_t quantType, double dstTypeMax, int64_t softmaxPrecision, "
            "int64_t winLeft, int64_t winRight, int64_t returnSoftmaxlse, "
            "const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )


class TestGenericBlockSparseAttentionTorch:
    @staticmethod
    def _torch_dtype_to_np(dtype):
        if dtype == torch.float16:
            return np.float16
        if dtype == torch.bfloat16:
            try:
                from ml_dtypes import bfloat16 as np_bf16
                return np_bf16
            except ImportError:
                return np.float32
        return np.float32

    @staticmethod
    def _tensor_to_np(t, np_dtype=None):
        t = safe_to_tensor(t)
        if t.dtype == torch.bfloat16:
            arr = t.detach().float().cpu().numpy()
            return arr.astype(np_dtype) if np_dtype is not None else arr
        arr = t.detach().cpu().numpy()
        if np_dtype is not None and arr.dtype != np_dtype:
            return arr.astype(np_dtype)
        return arr

    @classmethod
    def base_tile_mm(cls, left, right, mm_k_tile=128):
        """Match BSA950: K-split matmul, accumulate in fp32."""
        k_dim = left.shape[1]
        if k_dim <= mm_k_tile:
            return np.matmul(
                np.asarray(left, dtype=np.float32),
                np.asarray(right, dtype=np.float32),
            )
        res = None
        mm_k_loop = (k_dim + mm_k_tile - 1) // mm_k_tile
        for idx in range(mm_k_loop):
            sub_k = min(mm_k_tile, k_dim - idx * mm_k_tile)
            left_slice = np.asarray(
                left[:, idx * mm_k_tile: idx * mm_k_tile + sub_k], dtype=np.float32
            )
            right_slice = np.asarray(
                right[idx * mm_k_tile: idx * mm_k_tile + sub_k, :], dtype=np.float32
            )
            res_slice = np.matmul(left_slice, right_slice)
            res = res_slice if res is None else (res + res_slice)
        return res

    @classmethod
    def _rowsum_half_tree(cls, p):
        """Match AscendC WholeReduceSum<half>: pairwise add with half rounding each step.

        numpy np.sum(fp16) accumulates in fp32 then casts once; device half tree-reduce
        can differ by 1 ulp and shows up in LSE via log(gl).
        """
        x = p.astype(np.float32)
        while x.shape[1] > 1:
            if x.shape[1] % 2 == 1:
                tail = x[:, -1:]
                x = x[:, :-1]
            else:
                tail = None
            x = (x[:, 0::2] + x[:, 1::2]).astype(np.float16).astype(np.float32)
            if tail is not None:
                if x.shape[1] == 0:
                    x = tail
                else:
                    x = np.concatenate(
                        [x[:, :-1],
                         (x[:, -1:] + tail).astype(np.float16).astype(np.float32)],
                        axis=1,
                    )
        return x[:, 0].astype(np.float16).reshape(-1, 1)

    @classmethod
    def ref_flash(cls, query, key, value, scale, np_dtype, use_low_sm):
        """BSA950-style tiled flash. query [S1,D], key [D,S2], value [S2,D].

        softmaxPrecision=1: SM in query.dtype, scale rounded to SM (arch35 ToBfloat16/half).
        softmaxPrecision=0: fp32 SM + DownCastP before PV.
        Rescale stays fp32 (arch35 mixed path).
        """
        cur_kv_s = key.shape[1]
        kv_s_base_tile = 128
        interm_dtype_re = np.float32
        if use_low_sm:
            interm_dtype_sm = np_dtype
            scale_f = np.float32(np.asarray(scale, dtype=np_dtype))
        else:
            interm_dtype_sm = np.float32
            scale_f = np.float32(scale)

        gm = gl = go = None
        for kv_s_start in range(0, cur_kv_s, kv_s_base_tile):
            cur_kv_s_tile = min(kv_s_base_tile, cur_kv_s - kv_s_start)
            key_tile = key[:, kv_s_start: kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start: kv_s_start + cur_kv_s_tile, :]
            qk = cls.base_tile_mm(query, key_tile, 128)
            if use_low_sm:
                qk = qk.astype(interm_dtype_sm)
                qk = (np.asarray(qk, dtype=np.float32) * scale_f).astype(interm_dtype_sm)
            else:
                qk = qk * scale_f
            sim = qk.astype(interm_dtype_sm)
            lm = np.max(sim, axis=-1, keepdims=True).astype(interm_dtype_sm)
            is_first = kv_s_start == 0
            if is_first:
                hm = lm
                gm = lm.astype(interm_dtype_re)
                dm = np.ones_like(gm, dtype=interm_dtype_re)
            else:
                lm_f = lm.astype(interm_dtype_re)
                hm_f = np.maximum(gm, lm_f)
                dm = np.exp(gm - hm_f).astype(interm_dtype_re)
                gm = hm_f
                hm = hm_f.astype(interm_dtype_sm)
            p = np.exp((sim - hm).astype(interm_dtype_sm))
            if interm_dtype_sm == np.float16:
                ll = cls._rowsum_half_tree(p)
            else:
                ll = np.sum(p, axis=-1, keepdims=True).astype(interm_dtype_sm)
            p = p.astype(np_dtype)
            lo = cls.base_tile_mm(p, value_tile, 128).astype(interm_dtype_re)
            if is_first:
                gl = ll.astype(interm_dtype_re)
                go = lo
            else:
                gl = gl * dm + ll.astype(interm_dtype_re)
                go = go * dm + lo

        go = (go / gl).astype(np_dtype)
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32)
        return np.asarray(go, dtype=np.float32), np.asarray(lse, dtype=np.float32)

    @staticmethod
    def ref_attention(query, key, value, scale):
        """BSA950 fp32 真值: one-shot matmul + softmax."""
        s = np.matmul(
            np.asarray(query, dtype=np.float32),
            np.asarray(key, dtype=np.float32),
        ) * np.float32(scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        p = np.exp(s - row_max)
        row_sum = np.sum(p, axis=-1, keepdims=True)
        p = p / row_sum
        o = np.matmul(p, np.asarray(value, dtype=np.float32))
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1).astype(np.float32)
        return o.astype(np.float32), lse

    def calc_data(
        self,
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        block_table,
        block_shape,
        q_seqlens,
        kv_seqlens,
        scale_value,
        softmax_precision,
        is_benchmark=False,
        q_storage_seqlens=None,
    ):
        """Return (out_bm, out_gt, lse_bm, lse_gt). Only the requested path is computed.

        is_benchmark=False -> kernel-aligned bm; True -> fp32 gt.
        """
        query_t = safe_to_tensor(query)
        np_dtype = self._torch_dtype_to_np(query_t.dtype)
        query_np = self._tensor_to_np(query_t, np_dtype)
        key_np = self._tensor_to_np(key, np_dtype)
        value_np = self._tensor_to_np(value, np_dtype)
        sparse_idx = self._tensor_to_np(sparse_block_idx).astype(np.int32)
        sparse_cnt = self._tensor_to_np(sparse_block_count).astype(np.int32)
        block_table_np = self._tensor_to_np(block_table).astype(np.int32)

        block_shape_x = int(block_shape[0])
        block_shape_y = int(block_shape[1])
        if block_shape_x != 1:
            raise ValueError(
                f"calc_data currently requires blockShapeX=1, got {block_shape_x}"
            )
        if q_storage_seqlens is None:
            q_storage_seqlens = q_seqlens
        batch = len(q_seqlens)
        total_q, num_heads, head_dim = query_np.shape
        kv_heads = int(sparse_idx.shape[0])
        group_size = num_heads // kv_heads
        top_k = int(sparse_idx.shape[2])
        use_low_sm = int(softmax_precision) == 1
        scale = float(scale_value)
        compute_bm = not is_benchmark
        compute_gt = is_benchmark

        out_bm = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        out_gt = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        lse_bm = np.zeros((total_q, num_heads, 1), dtype=np.float32)
        lse_gt = np.zeros((total_q, num_heads, 1), dtype=np.float32)

        q_offset = 0
        for b in range(batch):
            q_seqlen = int(q_seqlens[b])
            kv_seqlen = int(kv_seqlens[b])
            if q_seqlen == 0 or kv_seqlen == 0:
                q_offset += int(q_storage_seqlens[b])
                continue
            history_len = kv_seqlen - q_seqlen
            for q_token in range(q_seqlen):
                global_q = q_offset + q_token
                causal_bound = history_len + q_token
                for kv_h in range(kv_heads):
                    valid_topk = int(sparse_cnt[kv_h, global_q])
                    if valid_topk <= 0:
                        continue
                    valid_topk = min(valid_topk, top_k)
                    k_parts = []
                    v_parts = []
                    idx_row = sparse_idx[kv_h, global_q, :valid_topk]
                    for logical_id in idx_row:
                        logical_id = int(logical_id)
                        if logical_id < 0:
                            continue
                        block_begin = logical_id * block_shape_y
                        block_end = min(block_begin + block_shape_y, kv_seqlen)
                        effective_end = min(block_end, causal_bound + 1)
                        if effective_end <= block_begin:
                            continue
                        physical_id = int(block_table_np[b, logical_id])
                        if physical_id < 0:
                            continue
                        tile_len = effective_end - block_begin
                        k_parts.append(key_np[physical_id, :tile_len, kv_h, :].T)
                        v_parts.append(value_np[physical_id, :tile_len, kv_h, :])
                    if not k_parts:
                        continue
                    key_g = np.concatenate(k_parts, axis=1)
                    val_g = np.concatenate(v_parts, axis=0)
                    q_start = kv_h * group_size
                    q_group = query_np[global_q, q_start: q_start + group_size, :]
                    if compute_bm:
                        go, lse = self.ref_flash(
                            q_group, key_g, val_g, scale, np_dtype, use_low_sm
                        )
                        out_bm[global_q, q_start: q_start + group_size, :] = go
                        lse_bm[global_q, q_start: q_start + group_size, 0] = lse
                    if compute_gt:
                        go, lse = self.ref_attention(q_group, key_g, val_g, scale)
                        out_gt[global_q, q_start: q_start + group_size, :] = go
                        lse_gt[global_q, q_start: q_start + group_size, 0] = lse
            q_offset += int(q_storage_seqlens[b])

        return (
            torch.from_numpy(out_bm),
            torch.from_numpy(out_gt),
            torch.from_numpy(lse_bm),
            torch.from_numpy(lse_gt),
        )


@register("aclnn_genericblocksparseattention")
class GenericBlockSparseAttentionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(GenericBlockSparseAttentionApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(10)
        apply_init_tensors(input_data, device=self.device)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        key = input_data.kwargs["key"]
        value = input_data.kwargs["value"]
        sparse_block_idx = input_data.kwargs["sparseBlockIdx"]
        sparse_block_count = input_data.kwargs["sparseBlockCount"]
        block_table = input_data.kwargs["blockTableOptional"]
        block_shape = input_data.kwargs["blockShape"]
        scale_value = input_data.kwargs["scaleValue"]
        softmax_precision = input_data.kwargs["softmaxPrecision"]
        return_lse = int(input_data.kwargs.get("returnSoftmaxlse", 0))

        # Prefer rewritten cu + seqused; else equal-batch recovery from shapes.
        query_t = safe_to_tensor(query)
        decoded = decode_cu_and_seqused(
            input_data.kwargs.get("cuSeqLengthsQOptional"),
            input_data.kwargs.get("cuSeqLengthsKvOptional"),
            seqused_q=input_data.kwargs.get("sequsedQOptional"),
            seqused_kv=input_data.kwargs.get("sequsedKvOptional"),
            total_q=int(query_t.shape[0]) if query_t is not None else None,
        )
        if decoded is not None:
            q_seqlens, kv_seqlens, q_storage_seqlens, _kv_storage = decoded
        else:
            q_seqlens, kv_seqlens = recover_batch_seqlens(
                query_t,
                safe_to_tensor(sparse_block_idx),
                safe_to_tensor(block_table),
            )
            q_storage_seqlens = q_seqlens

        test_obj = TestGenericBlockSparseAttentionTorch()
        atten_out_bm, atten_out_gt, lse_bm, lse_gt = test_obj.calc_data(
            query,
            key,
            value,
            sparse_block_idx,
            sparse_block_count,
            block_table,
            block_shape,
            q_seqlens,
            kv_seqlens,
            scale_value,
            softmax_precision,
            is_benchmark=bool(self.task_result.is_benchmark_task),
            q_storage_seqlens=q_storage_seqlens,
        )

        # BSA tests/950: False -> kernel-aligned bm (标杆); True -> fp32 gt (真值)
        if not self.task_result.is_benchmark_task:
            atten_out = atten_out_bm
            lse_out = lse_bm
        else:
            atten_out = atten_out_gt
            lse_out = lse_gt

        if return_lse == 1:
            return atten_out, lse_out
        return atten_out
