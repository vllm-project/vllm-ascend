import torch
import torch_npu
import torchair
import numpy as np
import torch.nn as nn

from torch_npu.testing.testcase import TestCase, run_tests

import logging
from torchair import logger
from vllm_ascend.utils import enable_custom_op

logger.setLevel(logging.DEBUG)
torch._logging.set_logs(graph_code=True)

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

def unpackbits(x, dim=-1):
    # 将x转换为二进制表示，并填充到8位
    # x: 任意形状的uint8张量
    # dim: 展开的维度
    shape = x.shape
    # 在dim维度上扩展8倍
    # new_shape = list(shape[:dim]) + [8] + list(shape[dim:])
    # 将x转换为二进制字符串并填充
    x_np = x.cpu().numpy()
    unpacked = np.unpackbits(x_np, axis=dim)
    return torch.from_numpy(unpacked).to(x.device).to(dtype=torch.float16)

def stable_topk_1d(input, k, largest=True, stable_index_order='ascending'):
    """
    一维稳定 topk 实现
    """
    # 创建 (值, 索引) 对
    values_with_indices = [(val.item(), idx) for idx, val in enumerate(input)]
    
    # 定义排序函数
    if largest:
        if stable_index_order == 'ascending':
            values_with_indices.sort(key=lambda x: (-x[0], x[1]))
        else:
            values_with_indices.sort(key=lambda x: (-x[0], -x[1]))
    else:
        if stable_index_order == 'ascending':
            values_with_indices.sort(key=lambda x: (x[0], x[1]))
        else:
            values_with_indices.sort(key=lambda x: (x[0], -x[1]))
    
    # 提取结果
    values = torch.tensor([x[0] for x in values_with_indices[:k]])
    indices = torch.tensor([x[1] for x in values_with_indices[:k]])
    
    return values, indices

def torch_hamming_distance_all(hash_q_op, hash_k_op, block_table_op, seq_len_op, chunk_op, top_k_op, sink, recent):
    print("torch_hamming_distance_all")
    # hash q op
    batch, num_head, max_q_seqlen, head_dim = hash_q_op.shape
    #assert num_head % 8 == 0
    # hash k op
    num_blocks, num_kv_head, block_size, head_dim = hash_k_op.shape

    num_head_group = num_head // num_kv_head
    ds_top_k_ = None
    ds_top_k_idx_ = None
    # batch loop
    for per_batch in range(batch) :
        # num_kv_head loop
        for single_num_kv_head in range(num_kv_head) :
            hash_q_op_per_batch = hash_q_op[per_batch:(per_batch+1), single_num_kv_head * num_head_group : (single_num_kv_head + 1) * num_head_group, :, :]
            ds_top_k, sk_idx = torch_hamming_distance_topk(
                hash_q_op_per_batch,
                hash_k_op[:, single_num_kv_head:(single_num_kv_head+1), :, :], 
                block_table_op[per_batch:(per_batch+1), :], 
                seq_len_op[per_batch], 
                chunk_op[per_batch], 
                top_k_op[per_batch],
                sink,
                recent
            )
            # print(f"ds_top_k = {ds_top_k}")
            # print(f"sk_idx = {sk_idx}")
            if ds_top_k_ is None:
                ds_top_k_ = ds_top_k
            else:
                ds_top_k_ = torch.cat([ds_top_k_, ds_top_k], dim=-1)
            
            if ds_top_k_idx_ is None:
                ds_top_k_idx_ = sk_idx
            else:
                ds_top_k_idx_ = torch.cat([ds_top_k_idx_, sk_idx], dim=-1)

    assert ds_top_k_ is not None
    assert ds_top_k_idx_ is not None
    top_k = top_k_op.max()
    ds_top_k_r = ds_top_k_.reshape(batch, num_kv_head, 1, top_k)
    ds_top_k_idx_r = ds_top_k_idx_.reshape(batch, num_kv_head, top_k)

    #print(f"ds_top_k_r = {ds_top_k_r}")
    # print(ds_top_k_idx_r[:6,:,:,:130])
    return ds_top_k_idx_r

def torch_hamming_distance_topk(hash_q_op, hash_k_op, block_table_op, seq_len, chunk_size, top_k, sink, recent):
    """
        算子实现逻辑：
        hash_q_op: (1, num_head, max_q_seqlen, head_dim) 输入的hash_q_op [num_head, max_q_seqlen, head_dim]会按照累加的方式合并成 [1, max_q_seqlen, head_dim]
        hash_k_op: (num_blocks, 1, block_size, head_dim)
        block_table_op: （1, max_num_blocks_per_seq） 存放 hash_k_op num_blocks维度的索引，便于获取hash_k_op与hash_q_op做hamming距离计算
        seq_len：token长度
        chunk_size:
        top_k:
    """

    # 根据目前hamming算子实现，转成Bit做矩阵乘法，计算hamming距离
    # hash q op
    bacth, num_head, max_q_seqlen, head_dim = hash_q_op.shape
    # print(f'hash_q_op.shape = {hash_q_op.shape}')  # torch.Size([1, 16, 1, 72])
    assert max_q_seqlen == 1
    # hash k op
    num_blocks, num_kv_head, block_size, head_dim = hash_k_op.shape
    # print(f'hash_k_op.shape = {hash_k_op.shape}')  # torch.Size([245, 1, 128, 72])

    # 将查询张量转换为二进制表示（bit）目前单个元素类型uint8
    hash_q_op_unpack = unpackbits(hash_q_op, dim=-1)
    # print(f"hash_q_op_unpack.shape = {hash_q_op_unpack.shape}")  # torch.Size([1, 16, 1, 576])
    hash_q_op_unpack = hash_q_op_unpack.squeeze(0).squeeze(1)    # torch.Size([16, 576])
    # print(f"hash_q_op_unpack.shape = {hash_q_op_unpack.shape}")
    # print(f"hash_q_op_unpack = {hash_q_op_unpack[0, :16]}")

    # 按照目前算子实现方式，对8*16的query按照bit位转换，bit位替换 1 -> 1 / 0 -> -1
    hash_q_op_unpack = hash_q_op_unpack.where(hash_q_op_unpack == 1, torch.tensor(-1, device='cpu')) # 1 -> 1 / 0 -> -1
    # print(f"hash_q_op_unpack.shape = {hash_q_op_unpack.shape}")
    # print(f"hash_q_op_unpack = {hash_q_op_unpack[0, :16]}")

    hash_q_op_unpack_cumsum = torch.cumsum(hash_q_op_unpack, dim=0)[-1] # 按行累加输出最后一行
    # print(f"hash_q_cumsum.shape = {hash_q_op_unpack_cumsum.shape}")  # torch.Size([576])
    # print(f"hash_q_cumsum = {hash_q_op_unpack_cumsum[:16]}")

    # 数据范围控制在[-8,8]
    if num_head > 8:
        div = (num_head + 7) // 8
        # print(f'div: {div}')
        reciprocalDiv = 1.0 / div
        # print(f'reciprocalDiv: {reciprocalDiv}, {type(reciprocalDiv)}')
        reciprocalDiv = torch.tensor((reciprocalDiv,), dtype=torch.float16, device='cpu')
        # print(f'reciprocalDiv: {reciprocalDiv}')
        hash_q_op_unpack_cumsum = hash_q_op_unpack_cumsum * reciprocalDiv
        # print(f"hash_q_cumsum = {hash_q_op_unpack_cumsum[:16]}")
    
    # 算子实现里面，由计算会将half cast 到int4b_t 当值达到8时，会cast到7
    hash_q_op_unpack_cumsum = torch.where(hash_q_op_unpack_cumsum == 8, torch.tensor(7, dtype=hash_q_op_unpack_cumsum.dtype, device='cpu'), hash_q_op_unpack_cumsum)
    print(f"hash_q_op_unpack_cumsum.shape = {hash_q_op_unpack_cumsum.shape}")  # torch.Size([576])
    # print(f"hash_q_op_unpack_cumsum = {hash_q_op_unpack_cumsum}")

    # 提取block_table_op中的非0元素
    block_table_op_origin = block_table_op
    block_table_op = block_table_op[block_table_op != 0]

    # 从hash_k_op中选出block_table_op位置的hashk数据
    key_selected = hash_k_op[block_table_op,:,:,:]
    # print(f"key_selected.shape = {key_selected.shape}")  # torch.Size([240, 1, 128, 72])

    # 将数据库张量转换为二进制表示（bit）
    key_selected_unpack = unpackbits(key_selected, dim=-1)
    # print(f"key_selected_unpack.shape = {key_selected_unpack.shape}")  # torch.Size([240, 1, 128, 576])
    # print(f"key_selected_unpack = {key_selected_unpack[0,0,0,:16]}")

    # 同样将key按照bit位转换，1 -> 1 / 0 -> -1
    key_selected_unpack = key_selected_unpack.where(key_selected_unpack == 1, torch.tensor(-1, device='cpu')) # 1 -> 1 / 0 -> -1
    # print(f"key_selected_unpack = {key_selected_unpack[0,0,0,:16]}")

    # q与K的乘积
    q_k_mul = torch.matmul(key_selected_unpack, hash_q_op_unpack_cumsum)
    print(f"q_k_mul.shape = {q_k_mul.shape}")  # torch.Size([240, 1, 128])
    print(f"q_k_mul = {q_k_mul[0,0,:]}")

    # 展开成一维
    flat_q_k_mul = q_k_mul.view(-1) # (1, 33 * 16)
    # 按照chunkSize 分段，每chunkSize中求取一个最大值

    last_len = seq_len % chunk_size
    effect_len = seq_len
    if last_len != 0:
        effect_len = (seq_len // chunk_size + 1) * chunk_size

    # chunk数
    chunk_blocks = effect_len // chunk_size
    # 取effect_len长度的token,按照chunk_size分割，并取最大值
    reduce_max_chunk = flat_q_k_mul[:effect_len].view(1, -1, chunk_size).max(dim=-1).values
    print(f"reduce_max_chunk = {reduce_max_chunk}")

    # sink
    reduce_max_chunk[:, :sink] = 8192
    # skip_tail_size
    chunk_block_size = reduce_max_chunk.shape[1]
    print(f"chunk_block_size = {chunk_block_size}")
    # todo 算子目前为了高效计算，使用Duplicate赋值数据，但是存在32字节对齐问题，导致最后尾部保留数据不准确
    # 在golden中要去模拟实现比较困难
    # 后续如果的确需要，再考虑实现
    reduce_max_chunk[:, chunk_block_size - recent: chunk_block_size] = 8192
    #print(f"reduce_max_chunk = {reduce_max_chunk}")
    # 求topk，并保证输出的indices有序
    topk_values, topk_indices = stable_topk_1d(reduce_max_chunk.view(-1), k=top_k, largest=True)
    #print(f"topk_values_new = {topk_values}")
    #print(f"topk_indices_new = {topk_indices}")
    # 对topk_indices 排序
    topk_indices_new_sort, idx = torch.sort(topk_indices)
    # print(f"topk_indices_new_sort = {topk_indices_new_sort}")
    # print(f"block_table_op_origin = {block_table_op_origin}")
    # 映射block_table_op的值
    block_table_indices = block_table_op_origin.view(-1)[topk_indices_new_sort]
    # print(f"block_table_indices = {block_table_indices}")
    
    return topk_values, block_table_indices

class TestCustomHammingDistTopK(TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.batch_size = 5
        self.num_head = 16
        self.num_kv_head = 1
        self.head_dim = 128
        self.compress_rate = 8
        self.compressed_dim = self.head_dim // self.compress_rate
        self.seqlen_q = 1
        self.sparse_ratio = 0.2
        self.chunk_size_value = 128
        self.device_id = 0
        self.DEVICE_ID = 0
        
        self.seqlen_list = [30720] * self.batch_size
        self.seqlen = torch.tensor(self.seqlen_list, dtype=torch.int32, device=self.device)
        self.max_seq_len = max(self.seqlen_list)
        self.chunk_size = torch.tensor([self.chunk_size_value] * self.batch_size, 
                                      dtype=torch.int32, device=self.device)
        self.top_k_list = [int(seq * self.sparse_ratio // self.chunk_size_value) 
                         for seq in self.seqlen_list]
        self.top_k = torch.tensor(self.top_k_list, dtype=torch.int32, device=self.device)
        print("self.top_k_list", self.top_k_list, self.top_k)
        self.block_size = 128
        self.num_blocks_per_seq = (self.seqlen + self.block_size - 1) // self.block_size
        self.num_blocks = self.num_blocks_per_seq.sum().item() + 5
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.qhash = torch.randint(255, 
                                  (self.batch_size, self.num_head, self.seqlen_q, self.compressed_dim), 
                                  dtype=torch.uint8, device=self.device)
        self.khash = torch.randint(255, 
                                  (self.num_blocks, self.num_kv_head, self.block_size, self.compressed_dim), 
                                  dtype=torch.uint8, device=self.device)
        self.sink = 1
        self.recent = 4

        # 初始化block_table
        max_num_blocks_per_seq = (max(self.seqlen_list) + self.block_size - 1) // self.block_size + 5
        self.block_table = torch.full((len(self.num_blocks_per_seq), max_num_blocks_per_seq), 
                                     fill_value=0, dtype=torch.int32)
        start = 1
        for i, n in enumerate(self.num_blocks_per_seq):
            self.block_table[i, :n] = torch.arange(start, start + n, dtype=torch.int32)
            start += n
        self.block_table = self.block_table.to(device=self.device)
        self.indices = torch.zeros([self.batch_size, self.num_kv_head, 128], dtype=torch.int32)

        self.support_offload = 1
        self.mask = torch.tensor([True, True, False, False, False])
        
        torch.npu.set_device(self.device_id)
        self.npu = f'npu:{self.device_id}'

    def _run_eager_mode_without_support_offload_and_mask(self):
        """运行单算子模式"""
        output_eager = torch.ops._C_ascend.npu_hamming_dist_top_k(
            self.qhash.to(self.npu), self.khash.to(self.npu), None, self.top_k.to(self.npu), 
            self.seqlen.to(self.npu), self.chunk_size.to(self.npu), self.max_seq_len, 
            self.sink, self.recent, None, self.block_table.to(self.npu), 
            None, self.indices.to(self.npu)
        )
        return output_eager
    def _run_eager_mode_with_support_offload_and_mask(self):
        """运行单算子模式"""
        output_eager = torch.ops._C_ascend.npu_hamming_dist_top_k(
            self.qhash.to(self.npu), self.khash.to(self.npu), None, self.top_k.to(self.npu), 
            self.seqlen.to(self.npu), self.chunk_size.to(self.npu), self.max_seq_len, 
            self.sink, self.recent, self.support_offload, self.block_table.to(self.npu), 
            self.mask.to(self.npu), self.indices.to(self.npu)
        )
        return output_eager

    def _run_graph_mode(self):
        """运行图模式"""
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, qhash, khash, khash_rope, top_k, seqlen, chunk_size, max_seq_len, 
                       sink, recent, support_offload, block_table, mask, indices):
                return torch.ops._C_ascend.npu_hamming_dist_top_k(
                    qhash, khash, None, top_k, seqlen, chunk_size, max_seq_len, 
                    sink, recent, support_offload, block_table, mask, indices
                )

        npu_mode = Network().to(f"npu:{self.DEVICE_ID}")
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        
        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        npu_out = npu_mode(
            self.qhash.to(self.npu), self.khash.to(self.npu), None, self.top_k.to(self.npu), 
            self.seqlen.to(self.npu), self.chunk_size.to(self.npu), self.max_seq_len, 
            self.sink, self.recent, self.support_offload, self.block_table.to(self.npu), 
            self.mask.to(self.npu), self.indices.to(self.npu)
        )
        return npu_out

    def test_hamming_dist_top_k_compare(self):
        """比较单算子模式和图模式的结果"""
        # 获取单算子模式的结果
        output_eager = self._run_eager_mode_with_support_offload_and_mask()
        
        # 获取图模式的结果
        output_graph = self._run_graph_mode()
        
        # 断言两种模式的结果相等
        print(f"===========output_eager {output_eager} ===================")
        print(f"===========output_graph {output_graph} ===================")
        self.assertEqual(output_eager.shape, output_graph.shape)
        self.assertTrue(torch.allclose(output_eager.float(), output_graph.float(), atol=1e-05))

    def test_hamming_dist_top_k(self, loop: int = 1, enable_assert: bool = True):
        output_gd = torch_hamming_distance_all(self.qhash, self.khash, self.block_table, self.seqlen, self.chunk_size, self.top_k, self.sink, self.recent)
        w = output_gd.shape[-1]
        print(f"output_gd shape: {output_gd.shape}")
        print(f"output_gd: {output_gd[:,:,:w]}")

        output_op = self._run_eager_mode_without_support_offload_and_mask()
        print(f"output_op shape: {output_op.shape}")
        print(f"output_op: {output_op[:,:,:w]}")
        assert torch.equal(output_op[:,:,:w].to('cpu'), output_gd[:,:,:w]), "Output from custom op does not match ground truth!"

if __name__ == "__main__":
    run_tests()