import gc
import math
import copy
import numpy as np
import torch
import torch_npu

# enable vllm-ascend custom ops
from vllm_ascend.utils import enable_custom_op
enable_custom_op()


def build_decay(head_num):
    # return decay rate with shape (head_num)
    start = 2 ** (-(2 ** -(math.log2(head_num) - 3)))
    ratio = start
    return torch.tensor([start * ratio**i for i in range(head_num)])


def lightning_attention_prefill(qt, kt, vt, kvsum, diag_decay, q_decay, block_decay, k_decay, dtype):
    # O_intra = [(Q_t K_t^T) * M]V_t
    qt_kt = torch.matmul(qt, torch.transpose(kt, 0, 1))
    qt_kt_mask = torch.mul(qt_kt, diag_decay).to(dtype)
    o_intra = torch.matmul(qt_kt_mask.to(torch.float32), vt)

    # O_inter = Λ Q_t (KV)
    o_inter = q_decay * torch.matmul(qt, kvsum.to(dtype).to(torch.float32))

    # update KVsum
    # KVsum = λ^B KVsum + (λ^B Λ^-1 K_t)^T V_t
    kt = k_decay * kt
    kt = kt.to(dtype)
    kt_vt = torch.matmul(torch.transpose(kt, 0, 1).to(torch.float32), vt)
    kvsum = torch.add(block_decay * kvsum, kt_vt)

    # O_t = O_intra + O_inter
    o_t = torch.add(o_intra, o_inter)

    return o_t, kvsum


def reference_lightning_attention(q, k, v, ed, block_size, kv_history, seq_len):
    dtype = q.dtype
    batch_num, head_num, n, d = q.shape
    if seq_len is None:
        seq_len = [n] * batch_num
    B = block_size
    T = n // B

    # get Q, K, V, decay
    # in_tensors[0]: Query without tiling (batch, head, n, d)
    # in_tensors[1]: Key without tiing (batch, head, n, d)
    # in_tensors[2]: Value without tiling (batch, head, n, d)
    # in_tensors[3]: Decay (head)
    query = q.reshape(batch_num, head_num, T, B, d).to(torch.float32) # (batch, head, T, B, d)
    key = k.reshape(batch_num, head_num, T, B, d).to(torch.float32)   # (batch, head, T, B, d)
    value = v.reshape(batch_num, head_num, T, B, d).to(torch.float32) # (batch, head, T, B, d)
    decay = ed.to(torch.float32)                                       # (head)

    # initialize O, KVsum
    output = torch.zeros(batch_num, head_num, T, B, d, dtype=dtype)          # (batch, head, T, B, d)
    if kv_history is None:
        kvsums = torch.zeros(batch_num, head_num, d, d, dtype=torch.float32)
    else:
        kvsums = kv_history.clone().to(torch.float32)              # (batch, head, d, d)

    for batchidx in range(batch_num):
        for headidx in range(head_num):
            kvsum = kvsums[batchidx, headidx, :, :]

            # diag_decay: M with shape (B, B)
            # q_decay: Λ with shape (B, 1)
            # block_decay: λ^B with shape (1)
            # k_decay: λ^B Λ^-1 with shape (B, 1)
            s = decay[headidx]
            i = torch.arange(B).view(B, 1)
            j = torch.arange(B)
            index = i - j
            diag_decay = torch.exp(s * torch.where(index>=0, -index, float('-inf')))
            q_decay = torch.exp(-s * (j + 1)).reshape(B, 1)
            block_decay = math.exp(-s * B)
            k_decay = torch.exp(-s * (B - i - 1))

            block_count = (seq_len[batchidx] + B - 1) // B
            tail_block_size = seq_len[batchidx] % B
            for t in range(block_count):
                qt = query[batchidx, headidx, t, :, :]
                kt = key[batchidx, headidx, t, :, :]
                vt = value[batchidx, headidx, t, :, :]
                if tail_block_size != 0 and t + 1 == block_count:
                    e = tail_block_size - i - 1
                    e[tail_block_size:] = 0
                    k_decay = torch.exp(-s * e)
                    block_decay = math.exp(-s * tail_block_size)
                o_t, kvsum = lightning_attention_prefill(
                    qt, kt, vt, kvsum, diag_decay, q_decay, block_decay, k_decay, dtype)
                output[batchidx, headidx, t, :, :] = o_t.to(dtype)

            kvsums[batchidx, headidx, :, :] = kvsum

    output = output.reshape(batch_num, head_num, n, d)          # (batch, head, n, d)
    kvsums = kvsums.to(dtype)
    return [output, kvsums]


def execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size,
                                                has_kv_history=False, actual_seq_len=None, dtype=torch.float16,
                                                slope_rate=None):

    base = 0.1
    query_cpu = base * torch.randn(batch_size, head_num, max_seq_len, head_dim).to(dtype)
    key_cpu   = base * torch.randn(batch_size, head_num, max_seq_len, head_dim).to(dtype)
    value_cpu = base * torch.randn(batch_size, head_num, max_seq_len, head_dim).to(dtype)
    if actual_seq_len:
        for b in range(batch_size):
            if actual_seq_len[b] < max_seq_len:
                query_cpu[b,:, actual_seq_len[b]:,:] = 0
                key_cpu[b,:, actual_seq_len[b]:,:] = 0
                value_cpu[b,:, actual_seq_len[b]:,:] = 0

    slope_rate_cpu = slope_rate
    if slope_rate_cpu is None:
        slope_rate_cpu = build_decay(head_num).to(dtype)

    query_npu = copy.deepcopy(query_cpu).npu()
    key_npu = copy.deepcopy(key_cpu).npu()
    value_npu = copy.deepcopy(value_cpu).npu()
    slope_rate_npu = copy.deepcopy(slope_rate_cpu).npu()
    kv_history_cpu = None
    kv_history_npu = None
    if has_kv_history:
        kv_history_cpu = base * torch.randn(batch_size, head_num, head_dim, head_dim).to(dtype)
        kv_history_npu = copy.deepcopy(kv_history_cpu).npu()

    # calculate on npu
    attention_npu_out, kv_cache_npu_out = torch.ops._C_ascend.npu_lightning_attention_prefill(
        query_npu, key_npu, value_npu, slope_rate_npu, block_size, kv_history_npu, actual_seq_len)

    # calculate on cpu
    attention_cpu_out, kv_cache_cpu_out = reference_lightning_attention(
        query_cpu, key_cpu, value_cpu, slope_rate_cpu, block_size, kv_history_cpu, actual_seq_len)

    if actual_seq_len:
        for b in range(batch_size):
            if actual_seq_len[b] < max_seq_len:
                # npu default value may not be 0
                attention_npu_out[b,:, actual_seq_len[b]:,:] = 0

    # compare result
    torch.testing.assert_close(attention_npu_out.cpu(),
                               attention_cpu_out,
                               atol=1e-3,
                               rtol=1e-3)
    torch.testing.assert_close(kv_cache_npu_out.cpu(),
                               kv_cache_cpu_out,
                               atol=1e-3,
                               rtol=1e-3)


@torch.inference_mode()
def test_lightning_attention_prefill_pad():
    batch_size = 1
    head_num = 4
    max_seq_len = 8192
    head_dim = 128
    block_size = 128
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_prefill_unpad_1():
    batch_size = 1
    head_num = 8
    max_seq_len = 16
    block_size = 16
    head_dim = 128
    actual_seq_len = [5]
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size, False,
                                                    actual_seq_len)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
def test_lightning_attention_prefill_unpad_2():
    batch_size = 4
    head_num = 8
    max_seq_len = 2048
    block_size = 128
    head_dim = 128
    actual_seq_len = [np.random.randint(1, max_seq_len / block_size + 1) * block_size
                        for _ in range(batch_size)]
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size,
                                                    False, actual_seq_len)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_prefill_unpad_3():
    batch_size = 3
    head_num = 8
    max_seq_len = 384
    block_size = 128
    head_dim = 128
    actual_seq_len = [351, 129, 384]
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size, False,
                                                    actual_seq_len)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_prefill_unpad_4():
    batch_size = 1
    head_num = 4
    max_seq_len = 256
    block_size = 256
    head_dim = 128
    actual_seq_len = [5]
    slope_rate = torch.tensor([0.9170, 0.8409, 0.7711, 0.7071], dtype=torch.float16)
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size, False,
                                                    actual_seq_len, torch.float16, slope_rate)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_prefill_with_kv_history():
    batch_size = 4
    head_num = 8
    max_seq_len = 1024
    head_dim = 128
    block_size = 128
    actual_seq_len = [np.random.randint(1, max_seq_len / block_size + 1) * block_size
                        for _ in range(batch_size)]
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size,
                                                    True, actual_seq_len)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_prefill_fp32():
    batch_size = 1
    head_num = 16
    max_seq_len = 256
    head_dim = 128
    block_size = 128
    actual_seq_len = [130]
    execute_lightning_attention_prefill_case(batch_size, head_num, max_seq_len, head_dim, block_size,
                                                    True, actual_seq_len, torch.float32)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
