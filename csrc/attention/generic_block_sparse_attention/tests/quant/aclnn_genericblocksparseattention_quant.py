"""GBSA quantType=5 (full FP8) ATK plugin — align BSA quant layout (bm+gt inline).

Self-contained: local metadata + sparse/cu/seqused init (no import from tests/st).
"""

import ctypes
import random
from dataclasses import dataclass

import numpy as np
import torch
from ml_dtypes import bfloat16

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, TORCH_TO_ACLTYPE

# ATK acl_wrapper lacks fp8 mapping; CANN DT_FLOAT8_E4M3FN = 36
TORCH_TO_ACLTYPE.setdefault("torch.float8_e4m3fn", 36)

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


def recover_batch_seqlens(query, sparse_block_idx, block_table, metadata=None):
    """Equal-batch seqlens: q=T/B, kv=maxBlocks*128."""
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
    np.random.seed(seed)
    random.seed(seed)
    batch = len(q_seqlens)
    if q_storage_seqlens is None:
        q_storage_seqlens = q_seqlens
    total_q_blocks = sum(int(s) for s in q_storage_seqlens)
    max_blocks = max((kv + block_shape_y - 1) // block_shape_y for kv in kv_seqlens)
    top_k = min(top_k, max_blocks)
    smoke = max_blocks == 1 and top_k == 1

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


def apply_init_tensors(input_data, device="cpu"):
    """Rewrite cuSeq / sparse / blockTable / seqused / metadata; keep QKV as generated."""
    query = safe_to_tensor(input_data.kwargs["query"])
    key = safe_to_tensor(input_data.kwargs["key"])
    sparse_block_idx = safe_to_tensor(input_data.kwargs["sparseBlockIdx"])
    block_table = safe_to_tensor(input_data.kwargs["blockTableOptional"])

    storage_q, storage_kv = recover_batch_seqlens(query, sparse_block_idx, block_table)
    kv_heads = int(sparse_block_idx.shape[0])
    top_k = int(sparse_block_idx.shape[2])
    num_physical = int(key.shape[0])
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

    return actual_q, actual_kv

# Kernel (arch35 full-quant bf16): FusedExpSub is fp32; ll = sum(fp32 exp).
SIMULATE_SM_EXP = False


def _to_fp32_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _numpy_to_torch(arr):
    if arr is None:
        return None
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


class TestGenericBlockSparseAttentionGenBmGt:
    @dataclass
    class AuxAttrs:
        num_heads: int
        kv_heads: int
        head_dim: int
        scale: float
        block_shape_y: int
        top_k: int
        sm_dtype: any = bfloat16  # attentionOut dtype: fp16 or bf16

    @dataclass
    class AttentionInputs:
        query: any
        key: any
        value: any
        sparse_block_idx: any
        sparse_block_count: any
        block_table: any
        q_seqlens: any
        kv_seqlens: any
        aux_attrs: any
        q_storage_seqlens: any = None

    @staticmethod
    def _fp32_to_sm(x, sm_dtype):
        return np.array(x, dtype=sm_dtype).astype(np.float32)

    @staticmethod
    def _fp32_to_bf16(x):
        return np.array(x, dtype=bfloat16).astype(np.float32)

    @staticmethod
    def _sm_dtype_from_attention_out(attention_out):
        t = safe_to_tensor(attention_out) if attention_out is not None else None
        if t is not None and t.dtype in (torch.float16, torch.half):
            return np.float16
        return bfloat16

    @classmethod
    def _float32_to_fp8_bits(cls, arr):
        max_fp8 = 448.0
        min_normal = 2.0 ** (-6)
        sign = (arr < 0).astype(np.uint8)
        abs_val = np.abs(arr)
        result_bits = np.zeros_like(arr, dtype=np.uint8)
        overflow_mask = abs_val >= max_fp8
        if np.any(overflow_mask):
            result_bits[overflow_mask] = np.where(sign[overflow_mask] == 0, 0x7E, 0xFE)
        zero_mask = arr == 0.0
        result_bits[zero_mask] = 0x00
        flat = arr.flatten()
        flat_abs = abs_val.flatten()
        flat_sign = sign.flatten()
        flat_bits = result_bits.flatten()
        valid = ~(overflow_mask.flatten() | zero_mask.flatten())
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return result_bits.reshape(arr.shape)
        v_abs = flat_abs[valid_idx]
        v_sign = flat_sign[valid_idx]
        under = v_abs < min_normal
        if np.any(under):
            ui = valid_idx[under]
            u_abs = v_abs[under]
            u_sign = v_sign[under]
            mant = np.clip(np.rint(u_abs / min_normal * 8.0), 0, 7).astype(np.uint8)
            flat_bits[ui] = (u_sign << 7) | mant
        normal = ~under
        if np.any(normal):
            ni = valid_idx[normal]
            n_abs = v_abs[normal]
            n_sign = v_sign[normal]
            _, exp_int = np.frexp(n_abs)
            exponent = np.clip(exp_int - 1, -6, 8)
            mant_float = n_abs / (2.0 ** exponent.astype(np.float32)) - 1.0
            mantissa = np.rint(mant_float * 8.0).astype(np.int32)
            carry = mantissa == 8
            mantissa[carry] = 0
            exponent[carry] += 1
            overflow_carry = exponent > 8
            if np.any(overflow_carry):
                co_ni = ni[overflow_carry]
                co_sign = n_sign[overflow_carry]
                flat_bits[co_ni] = np.where(co_sign == 0, 0x7E, 0xFE)
                keep = ~overflow_carry
                if np.any(keep):
                    ni = ni[keep]
                    n_sign = n_sign[keep]
                    mantissa = mantissa[keep]
                    exponent = exponent[keep]
                else:
                    ni = np.array([], dtype=np.int64)
                    mantissa = np.array([], dtype=np.int32)
                    exponent = np.array([], dtype=np.int32)
            if len(ni) > 0:
                mantissa = np.clip(mantissa, 0, 7).astype(np.uint8)
                stored_exp = np.clip(exponent + 7, 1, 15).astype(np.uint8)
                flat_bits[ni] = (n_sign << 7) | (stored_exp << 3) | mantissa
        return flat_bits.reshape(arr.shape)

    @classmethod
    def _fp8_bits_to_float32(cls, bits):
        sign = (bits >> 7) & 0x1
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        result = np.zeros_like(bits, dtype=np.float32)
        normal_mask = exponent != 0
        if np.any(normal_mask):
            exp_val = exponent[normal_mask].astype(np.float32)
            mant_val = mantissa[normal_mask].astype(np.float32)
            val = (2.0 ** (exp_val - 7)) * (1.0 + mant_val / 8.0)
            result[normal_mask] = np.where(sign[normal_mask] == 1, -val, val)
        subnormal = (exponent == 0) & (mantissa != 0)
        if np.any(subnormal):
            val = (2.0 ** (-6)) * (mantissa[subnormal].astype(np.float32) / 8.0)
            result[subnormal] = np.where(sign[subnormal] == 1, -val, val)
        return result

    @classmethod
    def _fp32_to_fp8_rint(cls, x):
        return cls._fp8_bits_to_float32(cls._float32_to_fp8_bits(x))

    @staticmethod
    def base_tile_mm(left, right, mm_k_tile=128):
        res = None
        k_dim = left.shape[1]
        for idx in range((k_dim + mm_k_tile - 1) // mm_k_tile):
            sub_k = min(mm_k_tile, k_dim - idx * mm_k_tile)
            a = left[:, idx * mm_k_tile: idx * mm_k_tile + sub_k].astype(np.float32)
            b = right[idx * mm_k_tile: idx * mm_k_tile + sub_k, :].astype(np.float32)
            s = np.matmul(a, b)
            res = s if res is None else res + s
        return res

    @classmethod
    def online_softmax(cls, qk_tile_res, gm, is_first_tile, interm_dtype_sm):
        sim = qk_tile_res.astype(interm_dtype_sm)
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first_tile:
            hm = lm
            dm = np.zeros_like(lm, dtype=np.float32)
        else:
            hm = np.maximum(gm, lm)
            dm = (gm - hm).astype(np.float32)
        gm = hm
        p = np.exp(sim.astype(np.float32) - hm.astype(np.float32))
        if SIMULATE_SM_EXP:
            p = cls._fp32_to_sm(p, interm_dtype_sm)
        ll = np.sum(p, axis=-1, keepdims=True)
        return p, ll, dm, gm

    @staticmethod
    def rescale_o(lo, ll, dm, go, gl, is_first_tile, interm_dtype_re):
        if is_first_tile:
            gl = ll
            go = lo
        else:
            dm_exp = np.exp(dm.astype(np.float32))
            gl = (gl * dm_exp + ll).astype(interm_dtype_re)
            go = (go * dm_exp.astype(interm_dtype_re) + lo).astype(interm_dtype_re)
        return go, gl.astype(interm_dtype_re)

    @classmethod
    def ref_flash_block_sparse_attention(cls, query, key, value, softmax_scale, block_lens,
                                         block_size=128, sm_dtype=bfloat16):
        # Align arch35 full-quant: SM QK in attentionOut dtype, ToBfloat16/half(scale),
        # fp32 exp+ll, P=CAST_RINT(fp8, exp*448) kept as fp32 for PV, final O CAST_RINT.
        cur_kv_len = key.shape[1]
        interm_dtype_sm = sm_dtype
        interm_dtype_re = np.float32
        scale_sm = cls._fp32_to_sm(np.asarray(softmax_scale, dtype=np.float32), interm_dtype_sm)
        gl = go = gm = None
        kv_start = 0
        for tile_idx, block_len in enumerate(block_lens):
            cur_tile = min(block_len, cur_kv_len - kv_start)
            if cur_tile <= 0:
                continue
            key_tile = key[:, kv_start: kv_start + cur_tile]
            value_tile = value[kv_start: kv_start + cur_tile, :]
            qk = cls.base_tile_mm(query, key_tile, 128)
            qk = cls._fp32_to_sm(qk, interm_dtype_sm)
            qk = cls._fp32_to_sm(qk * scale_sm, interm_dtype_sm)
            p, ll, dm, gm = cls.online_softmax(qk, gm, tile_idx == 0, interm_dtype_sm)
            p = cls._fp32_to_fp8_rint(p * 448.0)
            lo = cls.base_tile_mm(p, value_tile, 128)
            go, gl = cls.rescale_o(lo, ll, dm, go, gl, tile_idx == 0, interm_dtype_re)
            kv_start += cur_tile
        go = go / gl
        go = go * (1.0 / 448.0)
        return cls._fp32_to_sm(go, interm_dtype_sm)

    @staticmethod
    def ref_attention(query, key, value, softmax_scale):
        s = np.matmul(query.astype(np.float32), key.astype(np.float32)).astype(np.float32)
        s = s * np.float32(softmax_scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        s_sub = s - row_max
        s_sub = np.exp(s_sub)
        row_sum = np.sum(s_sub, axis=-1, keepdims=True)
        p = s_sub / row_sum
        o = np.matmul(p.astype(np.float32), value.astype(np.float32)).astype(np.float32)
        return o

    @staticmethod
    def gather_kv_blocks(key, value, batch_idx, kv_h, sparse_idx, block_table,
                         kv_seqlen, causal_bound, block_shape_y):
        key_parts, value_parts, block_lens = [], [], []
        for i in range(len(sparse_idx)):
            logical_id = int(sparse_idx[i])
            if logical_id < 0:
                continue
            block_begin = logical_id * block_shape_y
            block_end = min(block_begin + block_shape_y, kv_seqlen)
            effective_end = min(block_end, causal_bound + 1)
            if effective_end <= block_begin:
                continue
            physical_id = int(block_table[batch_idx, logical_id])
            if physical_id < 0:
                continue
            valid_len = effective_end - block_begin
            k_blk = _to_fp32_np(key[physical_id, :valid_len, kv_h, :]).T
            v_blk = _to_fp32_np(value[physical_id, :valid_len, kv_h, :])
            key_parts.append(k_blk)
            value_parts.append(v_blk)
            block_lens.append(valid_len)
        if not key_parts:
            return None, None, None
        return (
            np.concatenate(key_parts, axis=1),
            np.concatenate(value_parts, axis=0),
            block_lens,
        )

    def compute_output(self, attention_inputs: AttentionInputs, is_benchmark):
        query = _to_fp32_np(attention_inputs.query)
        key = attention_inputs.key
        value = attention_inputs.value
        sparse_block_idx = _to_fp32_np(attention_inputs.sparse_block_idx).astype(np.int32)
        sparse_block_count = _to_fp32_np(attention_inputs.sparse_block_count).astype(np.int32)
        block_table = _to_fp32_np(attention_inputs.block_table).astype(np.int32)
        q_seqlens = attention_inputs.q_seqlens
        kv_seqlens = attention_inputs.kv_seqlens
        q_storage_seqlens = attention_inputs.q_storage_seqlens
        if q_storage_seqlens is None:
            q_storage_seqlens = q_seqlens

        aux = attention_inputs.aux_attrs
        kv_heads = aux.kv_heads
        group_size = aux.num_heads // kv_heads
        scale = aux.scale
        block_shape_y = aux.block_shape_y
        top_k = aux.top_k
        sm_dtype = getattr(aux, "sm_dtype", bfloat16)

        total_q, num_heads, head_dim = query.shape
        if not is_benchmark:
            attn_out_bm = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        else:
            attn_out_gt = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)

        batch = len(q_seqlens)
        q_offset = 0
        for batch_idx in range(batch):
            q_seqlen = int(q_seqlens[batch_idx])
            kv_seqlen = int(kv_seqlens[batch_idx])
            if q_seqlen == 0 or kv_seqlen == 0:
                q_offset += int(q_storage_seqlens[batch_idx])
                continue
            history_len = kv_seqlen - q_seqlen
            for q_token in range(q_seqlen):
                global_q = q_offset + q_token
                global_q_block = global_q
                causal_bound = history_len + q_token
                for kv_h in range(kv_heads):
                    valid_topk = int(sparse_block_count[kv_h, global_q_block])
                    if valid_topk <= 0:
                        continue
                    valid_topk = min(valid_topk, top_k)
                    idx_row = sparse_block_idx[kv_h, global_q_block, :valid_topk]
                    key_g, val_g, block_lens = self.gather_kv_blocks(
                        key, value, batch_idx, kv_h, idx_row, block_table,
                        kv_seqlen, causal_bound, block_shape_y,
                    )
                    if key_g is None:
                        continue
                    q_start = kv_h * group_size
                    q_group = query[global_q, q_start: q_start + group_size, :]
                    if not is_benchmark:
                        out_group = self.ref_flash_block_sparse_attention(
                            q_group, key_g, val_g, scale, block_lens, block_shape_y,
                            sm_dtype=sm_dtype,
                        )
                        attn_out_bm[global_q, q_start: q_start + group_size, :] = out_group
                    else:
                        out_group = self.ref_attention(q_group, key_g, val_g, scale)
                        attn_out_gt[global_q, q_start: q_start + group_size, :] = out_group
            q_offset += int(q_storage_seqlens[batch_idx])

        if not is_benchmark:
            return attn_out_bm, None
        return None, attn_out_gt


@register("aclnn_genericblocksparseattentioninputprocess")
class GenericBlockSparseAttentionQuantInputProcess(AclnnBaseApi):
    """Quant InputProcess: patch fp8 dtype + reuse st sparse/cuSeq init."""

    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
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


@register("aclnn_genericblocksparseattention")
class GenericBlockSparseAttentionQuantApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(GenericBlockSparseAttentionQuantApi, self).__init__(task_result)

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
        return_lse = int(input_data.kwargs.get("returnSoftmaxlse", 0))

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

        sparse_t = safe_to_tensor(sparse_block_idx)
        kv_heads = int(sparse_t.shape[0])
        num_heads = int(query_t.shape[1])
        top_k = int(sparse_t.shape[2])

        test_obj = TestGenericBlockSparseAttentionGenBmGt()
        aux = test_obj.AuxAttrs(
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=int(query_t.shape[2]),
            scale=float(scale_value),
            block_shape_y=int(block_shape[1]),
            top_k=top_k,
            sm_dtype=test_obj._sm_dtype_from_attention_out(
                input_data.kwargs.get("attentionOut")
            ),
        )
        inputs = test_obj.AttentionInputs(
            query=query,
            key=key,
            value=value,
            sparse_block_idx=sparse_block_idx,
            sparse_block_count=sparse_block_count,
            block_table=block_table,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            aux_attrs=aux,
            q_storage_seqlens=q_storage_seqlens,
        )
        attn_out_bm, attn_out_gt = test_obj.compute_output(
            inputs, self.task_result.is_benchmark_task
        )

        if not self.task_result.is_benchmark_task:
            atten_out = _numpy_to_torch(attn_out_bm)
        else:
            atten_out = _numpy_to_torch(attn_out_gt)

        if return_lse == 1:
            lse = torch.zeros((query_t.shape[0], query_t.shape[1], 1), dtype=torch.float32)
            return atten_out, lse
        return atten_out
