import random
import numpy as np
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
import pytest
# torch.set_printoptions(threshold=np.inf)
import gc
enable_custom_op()
# num_tokens = 14
# num_head = 1
# head_size = 16
# block_size = 16
# num_blocks = 53535
def cal_reshape( key, value, key_cache, value_cache, slot_mapping,block_size):
    key_expect = key_cache.clone()
    value_expect = value_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        token_v = value[i]

        block_index = slot // block_size
        block_offset = slot % block_size
        key_expect[block_index][block_offset] = token_key
        value_expect[block_index][block_offset] = token_v
    return key_expect.npu(), value_expect.npu()   

def cal_slot( key, key_cache, slot_mapping,block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        block_index = slot // block_size
        block_offset = slot % block_size
        key_expect[block_index][block_offset] = token_key
    return key_expect.npu()

def cal_scatternd( key, key_cache, slot_mapping,block_size):
    key_expect = key_cache.clone()
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        token_key = key[i]
        key_expect[slot] = token_key

    return key_expect.npu()
 
 #c8 [ZTLOG] kv_cache[3].shape=torch.Size([423, 128, 1, 1]) slot_mapping.shape=torch.Size([16]) k_li_scale.shape=torch.Size([16, 1])
# [ZTLOG] kv_cache[3].view(-1, k_li_scale.shape[-1]).shape=torch.Size([54144, 1]) slot_mapping.view(-1, 1).shape=torch.Size([16, 1]) kv_cache[3].view(-1, k_li_scale.shape[-1]).shape=torch.Size([54144, 1])


#[ZTLOG] k_nope.shape=torch.Size([4, 1, 512]) k_pe.shape=torch.Size([4, 1, 64])
#  kv_cache[0].shape=torch.Size([385, 128, 1, 512]) kv_cache[1].shape=torch.Size([385, 128, 1, 64]) 
# slot_mapping[].shape=torch.Size([4]) 
@pytest.mark.parametrize('num_tokens', [8192])
@pytest.mark.parametrize('num_head', [1])
@pytest.mark.parametrize('block_size', [128])
@pytest.mark.parametrize('num_blocks', [461])
@pytest.mark.parametrize('count', [50])
def test_reshape_and_cache(num_tokens, num_head, block_size, num_blocks, count):
    # test 2 scatterndupdate with 1 resshapeandcache
    head_size_k = 512
    head_size_v = 64
    key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16).npu()
    value = torch.rand((num_tokens, num_head, head_size_v), dtype=torch.float16).npu()
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16).npu()
    value_cache = torch.rand((num_blocks, block_size, num_head, head_size_v), dtype=torch.float16).npu()
    slot_list=[]
    for i in range(0,num_tokens):
        # slot_list.append([6+i])
        slot_list.append(0+i)
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect, value_expect = cal_reshape(key, value, key_cache, value_cache, slot_mapping_npu,block_size)

    # slot_mapping = torch.from_numpy(slot_mapping).to(torch.int32).npu()
    # torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu.view(-1, 1), key)
    # torch_npu.npu_scatter_nd_update_(value_cache, slot_mapping_npu.view(-1, 1), value)
    torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping_npu)
    # torch_npu._npu_reshape_and_cache_siso(value, value_cache, slot_mapping_npu)
    # torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )
    torch.testing.assert_close(value_expect, value_cache, atol=0.001, rtol=0.1)

# slot_mapping[].shape=torch.Size([4]) 
@pytest.mark.parametrize('num_tokens', [32*1024])
@pytest.mark.parametrize('num_head', [1])
@pytest.mark.parametrize('block_size', [128])
@pytest.mark.parametrize('num_blocks', [461])
@pytest.mark.parametrize('count', [50])
def test_siso(num_tokens, num_head, block_size, num_blocks, count):
    # test 2 scatterndupdate with 1 resshapeandcache
    head_size_k = 512
    key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16).npu()
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16).npu()
    slot_list=[]
    for i in range(0,num_tokens):
        slot_list.append(0+i)
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_slot(key, key_cache, slot_mapping_npu,block_size)

    torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )

#   zt_kv0.shape=torch.Size([182272, 512]) slot_mapping.shape=torch.Size([16, 1]) k_nope.shape=torch.Size([16, 512])
#   zt_kv1.shape=torch.Size([182272, 64]) slot_mapping.shape=torch.Size([16, 1]) k_pe.shape=torch.Size([16, 64])
@pytest.mark.parametrize('num_tokens', [32*1024])#32*1024
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [461])#1599
@pytest.mark.parametrize('count', [1])
def test_scatter(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k =1
    key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16).npu()
    key_cache = torch.rand((num_blocks* block_size, num_head,head_size_k), dtype=torch.float16).npu()
    slot_list=[]
    for i in range(0,num_tokens):
        slot_list.append([0+i])
        # slot_list.append(6+i)
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_scatternd(key, key_cache, slot_mapping_npu,block_size)

    torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu, key)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )

    


@pytest.mark.parametrize('num_tokens', [16])#32*1024 6017
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [461])#1599
@pytest.mark.parametrize('count', [1])
def test_myops(num_tokens, num_head, block_size, num_blocks,count):
    head_size_k =512
    # key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16)
    # while count >0: 
    slot_list=[]
    for i in range(0,num_tokens):
        # slot_list.append([6+i])
        slot_list.append(8+i)
    # slot_list=[29572, 29236, 29406, 29542, 28872, 28023, 29098, 29184]
    # slot_list=[29572 ,29236 , 29184,  29406, 29542, 28872, 28023, 29098]
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16)
    key_npu = key.npu()
    # key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16)
    key_cache = torch.rand((num_blocks, block_size, num_head,head_size_k), dtype=torch.float16)
    key_cache_npu = key_cache.npu()

    key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)
    slot_mapping_list = slot_mapping_cpu.tolist() 
    group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.cache_by_group_pre( slot_mapping_npu, slot_mapping_list, block_size)
    torch.ops._C_ascend.reshape_and_cache_by_group(key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size)

    torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1 )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
        # count=count-1
