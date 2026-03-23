import gc
import math
import copy
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


def lightning_attention_decode(q, k, v, kv_decay, kv_cache, dtype):
    kv_cur = torch.outer(k, v)
    kv_pre = kv_decay * kv_cache
    kv = kv_cur + kv_pre
    o = torch.matmul(q, kv)
    return o, kv


def reference_lightning_attention_decode(query, key, value, slope_rate, kv_history, slot_ids, dtype):
    # in_tensors[0]: Query      (batch, head, 1, d)
    # in_tensors[1]: Key        (batch, head, 1, d)
    # in_tensors[2]: Value      (batch, head, 1, d)
    # in_tensors[3]: Decay      (head)
    # in_tensors[4]: KV Caches  (batch, head, d, d)
    # in_tensors[5]: slot_ids   (batch)
    batch_num, head_num, _, d = query.shape
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    slope_rate = slope_rate.to(torch.float32)
    kv_caches = kv_history.clone().to(torch.float32)

    # initialize O (batch, head * d)
    output = torch.zeros(batch_num, head_num * d, dtype=dtype)

    for batchidx in range(batch_num):
        slot_id = slot_ids[batchidx]
        for headidx in range(head_num):
            q = query[batchidx, headidx, 0, :]
            k = key[batchidx, headidx, 0, :]
            v = value[batchidx, headidx, 0, :]
            kv_decay = math.exp(-slope_rate[headidx])
            kv_cache = kv_caches[slot_id, headidx, :, :]
            o, kv = lightning_attention_decode(q, k, v, kv_decay, kv_cache, dtype)
            output[batchidx, headidx*d:(headidx+1)*d] = o.to(dtype)
            kv_caches[slot_id, headidx, :, :] = kv

    return output, kv_caches.to(dtype)


def execute_lightning_attention_decode_case(self, q_batch_size, kv_cache_batch, head_num, head_dim,
                                            dtype=torch.float16):
    query_cpu = torch.randn(q_batch_size, head_num, 1, head_dim).to(dtype)
    key_cpu = torch.randn(q_batch_size, head_num, 1, head_dim).to(dtype)
    value_cpu = torch.randn(q_batch_size, head_num, 1, head_dim).to(dtype)
    slope_rate_cpu = build_decay(head_num).to(dtype)
    kv_history_cpu = torch.randn(kv_cache_batch, head_num, head_dim, head_dim).to(dtype)
    slot_ids_cpu = torch.arange(kv_cache_batch).to(torch.int32)[-q_batch_size:]

    query_npu = copy.deepcopy(query_cpu).npu()
    key_npu = copy.deepcopy(key_cpu).npu()
    value_npu = copy.deepcopy(value_cpu).npu()
    slope_rate_npu = copy.deepcopy(slope_rate_cpu).npu()
    kv_history_npu = copy.deepcopy(kv_history_cpu).npu()
    slot_ids_npu = copy.deepcopy(slot_ids_cpu).npu()


    # calculate on npu
    attention_npu_out = torch.ops._C_ascend.npu_lightning_attention_decode(
        query_npu, key_npu, value_npu, kv_history_npu, slope_rate_npu, slot_ids_npu)

    # calculate on cpu
    attention_cpu_out, kv_cache_cpu_out = reference_lightning_attention_decode(
        query_cpu, key_cpu, value_cpu, slope_rate_cpu, kv_history_cpu, slot_ids_cpu, dtype)

    # compare result
    torch.testing.assert_close(attention_npu_out.cpu(),
                               attention_cpu_out,
                               atol=1e-9,
                               rtol=1e-6)
    torch.testing.assert_close(kv_history_npu.cpu(),
                               kv_cache_cpu_out,
                               atol=1e-9,
                               rtol=1e-6)


@torch.inference_mode()
def test_lightning_attention_decode_same_batch(self):
    q_batch_size = 256
    head_num = 8
    head_dim = 128
    execute_lightning_attention_decode_case(q_batch_size, q_batch_size, head_num, head_dim)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_decode_different_batch(self):
    q_batch_size = 1
    kv_cache_batch = 256
    head_num = 8
    head_dim = 128
    execute_lightning_attention_decode_case(q_batch_size, kv_cache_batch, head_num, head_dim)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@torch.inference_mode()
def test_lightning_attention_decode_fp32(self):
    q_batch_size = 100
    head_num = 16
    head_dim = 128
    execute_lightning_attention_decode_case(q_batch_size, q_batch_size, head_num, head_dim, torch.float32)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
