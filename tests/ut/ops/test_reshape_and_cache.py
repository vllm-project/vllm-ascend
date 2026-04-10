import random
import numpy as np
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
import pytest
torch.set_printoptions(threshold=np.inf)
import gc
import time
import os
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
@pytest.mark.parametrize('num_tokens', [16])
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
    # torch.ops._C_ascend.reshape_and_cache_by_group(key, key_cache, slot_list_np, block_size)
    # torch.ops._C_ascend.reshape_and_cache_by_group(value, value_cache, slot_list_np, block_size)
    torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping_npu)
    # torch_npu._npu_reshape_and_cache_siso(value, value_cache, slot_mapping_npu)
    # torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )
    torch.testing.assert_close(value_expect, value_cache, atol=0.001, rtol=0.1)

# slot_mapping[].shape=torch.Size([4]) 
@pytest.mark.parametrize('num_tokens', [32*1024])#6398
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [1773])#1599
@pytest.mark.parametrize('count', [1])
def test_siso(num_tokens, num_head, block_size, num_blocks, count):
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=[
    #         torch_npu.profiler.ExportType.Text,
    #         torch_npu.profiler.ExportType.Db
    #         ],
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    #     msprof_tx=False,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None,
    #     host_sys=[
    #         torch_npu.profiler.HostSystem.CPU,
    #         torch_npu.profiler.HostSystem.MEM],
    #     sys_io=False,
    #     sys_interconnection=False
    # )

    # prof = torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #         ],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/home/z00893411/ops/script/ls"),
    #     record_shapes=True,
    #     profile_memory=False,
    #     with_stack=True,
    #     with_modules=False,
    #     with_flops=False,
    #     experimental_config=experimental_config)

    # test 2 scatterndupdate with 1 resshapeandcache
    head_size_k = 512
    key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16).npu()
    # key = torch.randint(low=0,high=128,size=(num_tokens,head_size_k), dtype=torch.int8 )
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16).npu()
    # key_cache = torch.randint(low=0,high=128,size=(num_blocks, block_size, num_head,head_size_k), dtype=torch.int8 )
   
    slot_list=[]
    for i in range(0,num_tokens):
        slot_list.append(2+i)
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_slot(key, key_cache, slot_mapping_npu,block_size)
    
    N = 100
    # start = time.perf_counter()
    # for _ in range(N):
    # prof.start()
    warm_up=0
    for _ in range(warm_up):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    N = 100
    start = time.perf_counter()
    for _ in range(N):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    end = time.perf_counter()
    avg_ms = (end - start) / N * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")
    # prof.stop()
    # end = time.perf_counter()
    # avg_ms = (end - start) / N * 1000
    # print(f"python 耗时: {avg_ms:.4f} ms")
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )

#   zt_kv0.shape=torch.Size([182272, 512]) slot_mapping.shape=torch.Size([16, 1]) k_nope.shape=torch.Size([16, 512])
#   zt_kv1.shape=torch.Size([182272, 64]) slot_mapping.shape=torch.Size([16, 1]) k_pe.shape=torch.Size([16, 64])
@pytest.mark.parametrize('num_tokens', [32*1024])#6398
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [1773])#1599
@pytest.mark.parametrize('count', [1])
def test_scatter(num_tokens, num_head, block_size, num_blocks, count):
    head_size_k =64
    key = torch.randint(low=0,high=128,size=(num_tokens,num_head,head_size_k), dtype=torch.int8 ).npu()
    # key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16).npu()
    
    key_cache = torch.randint(low=0,high=128,size=(num_blocks*block_size, num_head,head_size_k), dtype=torch.int8 ).npu()
    # key_cache = torch.rand((num_blocks* block_size, num_head,head_size_k), dtype=torch.float16).npu()
    slot_list=[]
    for i in range(0,num_tokens):
        slot_list.append([2+i])
        # slot_list.append(6+i)
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = cal_scatternd(key, key_cache, slot_mapping_npu,block_size)
    N=100
    for i in range(N):
        torch_npu.npu_scatter_nd_update_(key_cache, slot_mapping_npu, key)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )


#   zt_kv0.shape=torch.Size([182272, 512]) slot_mapping.shape=torch.Size([16, 1]) k_nope.shape=torch.Size([16, 512])
#   zt_kv1.shape=torch.Size([182272, 64]) slot_mapping.shape=torch.Size([16, 1]) k_pe.shape=torch.Size([16, 64])
@pytest.mark.parametrize('num_tokens', [32*1024])#6398
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [1773])#1599
@pytest.mark.parametrize('count', [1])
def test_smalltokens(num_tokens, num_head, block_size, num_blocks,count):
    head_size_k =512
    
    #key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16)
    key = torch.randint(low=0,high=128,size=(num_tokens,head_size_k), dtype=torch.int8 )
    slot_list=[]
    for i in range(0,num_tokens):
        # slot_list.append(6+i)
        slot_list.append([6+i])
    # slot_list = [7,8,9,30,31,32,33,34,35,36,37,38,39,60,71,82]
    
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    # key = torch.rand((num_tokens, num_head), dtype=torch.float16)
    key_npu = key.npu()
    # key_cache = torch.rand((num_blocks, block_size, num_head, head_size_k), dtype=torch.float16)
    key_cache = torch.randint(low=0,high=128,size=(num_blocks, block_size, num_head,head_size_k), dtype=torch.int8 ).npu()
    # key_cache = torch.rand((num_blocks* block_size, num_head,head_size_k), dtype=torch.float16).npu()
    # key_cache = torch.rand((num_blocks, block_size, num_head), dtype=torch.float16)
    key_cache_npu = key_cache.npu()

    key_expect = cal_slot(key_npu, key_cache_npu, slot_mapping_npu, block_size)

    torch_npu.npu_scatter_nd_update_(key_cache_npu, slot_mapping_npu, key_npu)
    # torch.ops._C_ascend.reshape_and_cache_by_group(key_npu, key_cache_npu, slot_list_np, block_size)
    torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1 )

    # 清空NPU缓存（针对昇腾NPU，释放未回收的显存）
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    
group_list= [ 80,      0,  96816,    128,     80,  96896,    128,    208,  97024,
            128,    336,  97152,    128,    464,  97280,    128,    592,  97408,
            128,    720,  97536,    128,    848,  97664,    128,    976,  97792,
            128,   1104,  97920,    128,   1232,  98048,    128,   1360,  98176,
            128,   1488,  98304,    128,   1616,  98432,    128,   1744,  98560,
            128,   1872,  98688,    128,   2000,  98816,    128,   2128,  98944,
            128,   2256,  99072,    128,   2384,  99200,    128,   2512,  99328,
            128,   2640,  99456,    128,   2768,  99584,    128,   2896,  99712,
            128,   3024,  99840,    128,   3152,  99968,    128,   3280, 100096,
            128,   3408, 100224,    128,   3536, 100352,    128,   3664, 100480,
            128,   3792, 100608,    128,   3920, 100736,    128,   4048, 100864,
            128,   4176, 100992,    128,   4304, 101120,    128,   4432, 101248,
            128,   4560, 101376,    128,   4688, 101504,    128,   4816, 101632,
            128,   4944, 101760,    128,   5072, 101888,    128,   5200, 102016,
            128,   5328, 102144,    128,   5456, 102272,    128,   5584, 102400,
            128,   5712, 102528,    128,   5840, 102656,    128,   5968, 102784,
            128,   6096, 102912,    128,   6224, 103040,    128,   6352, 103168,
            128,   6480, 103296,    128,   6608, 103424,    128,   6736, 103552,
            128,   6864, 103680,    128,   6992, 103808,    128,   7120, 103936,
            128,   7248, 104064,    128,   7376, 104192,    128,   7504, 104320,
            128,   7632, 104448,    128,   7760, 104576,    128,   7888, 104704,
            128,   8016, 104832,    128,   8144, 104960,    128,   8272, 105088,
            128,   8400, 105216,    128,   8528, 105344,    128,   8656, 105472,
            128,   8784, 105600,    128,   8912, 105728,    128,   9040, 105856,
            128,   9168, 105984,    128,   9296, 106112,    128,   9424, 106240,
            128,   9552, 106368,    128,   9680, 106496,    128,   9808, 106624,
            128,   9936, 106752,    128,  10064, 106880,    128,  10192, 107008,
            128,  10320, 107136,    128,  10448, 107264,    128,  10576, 107392,
            128,  10704, 107520,    128,  10832, 107648,    128,  10960, 107776,
            128,  11088, 107904,    128,  11216, 108032,    128,  11344, 108160,
            128,  11472, 108288,    128,  11600, 108416,    128,  11728, 108544,
            128,  11856, 108672,    128,  11984, 108800,    128,  12112, 108928,
            128,  12240, 109056,    128,  12368, 109184,    128,  12496, 109312,
            128,  12624, 109440,    128,  12752, 109568,    128,  12880, 109696,
            128,  13008, 109824,    128,  13136, 109952,    128,  13264, 110080,
            128,  13392, 110208,    128,  13520, 110336,    128,  13648, 110464,
            128,  13776, 110592,    128,  13904, 110720,    128,  14032, 110848,
            128,  14160, 110976,    128,  14288, 111104,    128,  14416, 111232,
            128,  14544, 111360,    128,  14672, 111488,    128,  14800, 111616,
            128,  14928, 111744,    128,  15056, 111872,    128,  15184, 112000,
            128,  15312, 112128,    128,  15440, 112256,    128,  15568, 112384,
            128,  15696, 112512,    128,  15824, 112640,    128,  15952, 112768,
            128,  16080, 112896,    128,  16208, 113024,    128,  16336, 113152,
            128,  16464, 113280,    128,  16592, 113408,    128,  16720, 113536,
            128,  16848, 113664,    128,  16976, 113792,    128,  17104, 113920,
            128,  17232, 114048,    128,  17360, 114176,    128,  17488, 114304,
            128,  17616, 114432,    128,  17744, 114560,    128,  17872, 114688,
            128,  18000, 114816,    128,  18128, 114944,    128,  18256, 115072,
            128,  18384, 115200,    128,  18512, 115328,    128,  18640, 115456,
            128,  18768, 115584,    128,  18896, 115712,    128,  19024, 115840,
            128,  19152, 115968,    128,  19280, 116096,    128,  19408, 116224,
            128,  19536, 116352,    128,  19664, 116480,    128,  19792, 116608,
            128,  19920, 116736,    128,  20048, 116864,    128,  20176, 116992,
            128,  20304, 117120,    128,  20432, 117248,    128,  20560, 117376,
            128,  20688, 117504,    128,  20816, 117632,    128,  20944, 117760,
            128,  21072, 117888,    128,  21200, 118016,    128,  21328, 118144,
            128,  21456, 118272,    128,  21584, 118400,    128,  21712, 118528,
            128,  21840, 118656,    128,  21968, 118784,    128,  22096, 118912,
            128,  22224, 119040,    128,  22352, 119168,    128,  22480, 119296,
            128,  22608, 119424,    128,  22736, 119552,    128,  22864, 119680,
            128,  22992, 119808,    128,  23120, 119936,    128,  23248, 120064,
            128,  23376, 120192,    128,  23504, 120320,    128,  23632, 120448,
            128,  23760, 120576,    128,  23888, 120704,    128,  24016, 120832,
            128,  24144, 120960,    128,  24272, 121088,    128,  24400, 121216,
            128,  24528, 121344,    128,  24656, 121472,    128,  24784, 121600,
            128,  24912, 121728,    128,  25040, 121856,    128,  25168, 121984,
            128,  25296, 122112,    128,  25424, 122240,    128,  25552, 122368,
            128,  25680, 122496,    128,  25808, 122624,    128,  25936, 122752,
            128,  26064, 122880,    128,  26192, 123008,    128,  26320, 123136,
            128,  26448, 123264,    128,  26576, 123392,    128,  26704, 123520,
            128,  26832, 123648,    128,  26960, 123776,    128,  27088, 123904,
            128,  27216, 124032,    128,  27344, 124160,    128,  27472, 124288,
            128,  27600, 124416,    128,  27728, 124544,    128,  27856, 124672,
            128,  27984, 124800,    128,  28112, 124928,    128,  28240, 125056,
            128,  28368, 125184,    128,  28496, 125312,    128,  28624, 125440,
            128,  28752, 125568,    128,  28880, 125696,    128,  29008, 125824,
            128,  29136, 125952,    128,  29264, 126080,    128,  29392, 126208,
            128,  29520, 126336,    128,  29648, 126464,    128,  29776, 126592,
            128,  29904, 126720,    128,  30032, 126848,    128,  30160, 126976,
            128,  30288, 127104,    128,  30416, 127232,    128,  30544, 127360,
            128,  30672, 127488,    128,  30800, 127616,    128,  30928, 127744,
            128,  31056, 127872,    128,  31184, 128000,    128,  31312, 128128,
              4,  31440, 128256,    128,  31444, 128384,    128,  31572, 128512,
            128,  31700, 128640,    128,  31828, 128768,    128,  31956, 128896,
            128,  32084, 129024,    128,  32212, 129152,    128,  32340, 129280,
             92,  32468, 129408]

@pytest.mark.parametrize('num_tokens', [32*1024])#6398
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [1773])#1599
@pytest.mark.parametrize('count', [1])
def test_myops(num_tokens, num_head, block_size, num_blocks,count):
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=[
    #         torch_npu.profiler.ExportType.Text,
    #         torch_npu.profiler.ExportType.Db
    #         ],
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    #     msprof_tx=False,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None,
    #     host_sys=[
    #         torch_npu.profiler.HostSystem.CPU,
    #         torch_npu.profiler.HostSystem.MEM],
    #     sys_io=False,
    #     sys_interconnection=False
    # )

    # prof = torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #         ],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/home/z00893411/ops/script/ls"),
    #     record_shapes=True,
    #     profile_memory=False,
    #     with_stack=True,
    #     with_modules=False,
    #     with_flops=False,
    #     experimental_config=experimental_config)

    head_size_k =64
    # key_cache = torch.rand((num_blocks, block_size, num_head,head_size_k), dtype=torch.float16)
    key_cache = torch.randint(low=0,high=128,size=(num_blocks, block_size, num_head,head_size_k), dtype=torch.int8 )
    key_cache_npu = key_cache.npu()

    # key = torch.rand((num_tokens, num_head, head_size_k), dtype=torch.float16)
    # while count >0: 
    # num_tokens=num_tokens+200*zt_i
    slot_list=[]
    for i in range(0,num_tokens):
        # slot_list.append([6+i])
        slot_list.append(2+i)
    # slot_list=[29572, 29236, 29406, 29542, 28872, 28023, 29098, 29184]
    # slot_list=[29572 ,29236 , 29184,  29406, 29542, 28872, 28023, 29098]
    # slot_list.append(-1)
    # slot_list.append(-1)
    #assert num_tokens==len(slot_list)
    
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
    #slot_mapping_cpu = slot_mapping_npu.to("cpu",non_blocking=True)
    # num_draft_tensor = slot_mapping_npu.to("cpu", non_blocking=True)
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)

    # key = torch.rand((num_tokens, num_head,head_size_k), dtype=torch.float16)
    key = torch.randint(low=0,high=128,size=(num_tokens,head_size_k), dtype=torch.int8 )
    key_npu = key.npu()
    key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)



    time.sleep(0.1)
        
    # print(f"    test_myops            ")
    # for zt_i in range(100):
    # 调用你的算子


    slot_mapping_list = slot_mapping_cpu.tolist() 
    warm_up=0
    for _ in range(warm_up):

        group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.cache_by_group_pre( slot_mapping_npu, slot_mapping_list, block_size)
        torch.ops._C_ascend.reshape_and_cache_by_group(key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size)
    N = 100
    #group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.cache_by_group_pre( slot_mapping_npu, slot_mapping_list, block_size)
    start = time.perf_counter()
    for _ in range(N):

        # group_list_np= np.array(group_list)
        # prof.start()
        group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.cache_by_group_pre( slot_mapping_npu, slot_mapping_list, block_size)
        torch.ops._C_ascend.reshape_and_cache_by_group(key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size)
        # prof.stop()
    end = time.perf_counter()
    avg_ms = (end - start) / N * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    torch.equal(key_expect, key_cache_npu)   
    #torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1 )
    # # 转 MB 方便看
    # mb_size = byte_size / 1024 / 1024
    # print(f"    groupInfo Tensor 显存大小：{mb_size:.4f} MB")
    # # del groupInfo
    # # torch.npu.empty_cache()
    # # # 同步，保证算子执行完
    # torch.npu.synchronize()

    # # 每 100 次打印一次显存
    # if zt_i % 10 == 0:
    #     allocated = torch.npu.memory_allocated() / 1024 / 1024
    #     reserved = torch.npu.memory_reserved() / 1024 / 1024
    #         print(f"【第 {zt_i} 次】 allocated: {allocated:.4f} MB | reserved: {reserved:.4f} MB")
        
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
        # count=count-1

def test_npu_scatter_nd_update():
    loaded_dict = torch.load("/home/z00893411/ops/script/kv_cache_data.pt")
    slot_mapping = loaded_dict["value"].npu()
    kv0 = loaded_dict["kv0"].npu()
    k_nope_expect = loaded_dict["k_nope"].npu()
    k_nope=torch.zeros_like(k_nope_expect)
    # kv1 = loaded_dict["kv1"].npu()
    # k_pe_expect = loaded_dict["k_pe"].npu()
    torch_npu.npu_scatter_nd_update_(kv0, slot_mapping, k_nope)
    torch.testing.assert_close(k_nope_expect, k_nope, atol=0.001, rtol=0.1)


def test_dumptensor():
    kv_cache0_cpu = torch.load("/home/z00893411/ops/script/ds3.2_dump/tensor_dump/tensor_dump/kv_cache0_view_timestamp_20260305_105228.pt")
    kv_cache0_npu = kv_cache0_cpu.to("npu")
    # print(f"{kv_cache0_npu.shape=}")

    slot_mapping_cpu  = torch.load("/home/z00893411/ops/script/ds3.2_dump/tensor_dump/tensor_dump/slot_mapping_timestamp_20260305_105229.pt")
    slot_mapping_npu= slot_mapping_cpu.to("npu")
    slot_tensor_flat = slot_mapping_cpu.squeeze(dim=1) 
    slot_mapping_list = slot_tensor_flat.tolist()
    count =  [sub for sub in slot_mapping_list if sub != -1]
    count_np =np.array(count)

    # print(f"{slot_mapping_cpu.shape=} {len(count)=} {count=}")
    k_nope_cpu  = torch.load("/home/z00893411/ops/script/ds3.2_dump/tensor_dump/tensor_dump/k_nope_timestamp_20260305_105229.pt")
    k_nope_npu = k_nope_cpu.to("npu")
    k_nope_npu_act=k_nope_npu[0:len(count)]#herre
    # print(f"{k_nope_npu.shape=}")
    # torch.ops._C_ascend.write_cache_by_group_list(key, key_cache, slot_mapping_cpu, block_size)
    kv_cache0_npu_act=kv_cache0_npu.clone()
    print(f"{kv_cache0_npu.shape=} {slot_mapping_npu.shape=} {k_nope_npu.shape=}")
    # torch_npu.npu_scatter_nd_update_(kv_cache0_npu, slot_mapping_npu, k_nope_npu)
    block_size=128
    key_expect = cal_slot(k_nope_npu, kv_cache0_npu, slot_mapping_list,block_size)
    torch.ops._C_ascend.write_cache_by_group_list(k_nope_npu_act, kv_cache0_npu_act, count_np, block_size)
    torch.testing.assert_close(key_expect, kv_cache0_npu_act, atol=0.001, rtol=0.1 )
    # for i,slot in enumerate(count):
    #     print(f"{k_nope_npu[i].shape=} {kv_cache0_npu[count[i]].shape=}")
    #     torch.testing.assert_close(k_nope_npu[i], kv_cache0_npu[count[i]], atol=0.001, rtol=0.1 )
    # torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )

# test_dumptensor()