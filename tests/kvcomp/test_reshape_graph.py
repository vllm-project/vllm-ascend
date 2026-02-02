import torch
import torch_npu
#from torch_npu.testing.testcase import TestCase, run_tests
import copy
import numpy as np

import torchair
import torch.nn as nn
from vllm_ascend.utils import enable_custom_op
enable_custom_op()



from torch_npu.testing.testcase import TestCase, run_tests

torch._logging.set_logs(graph_code=True)

import logging
from torchair import logger
logger.setLevel(logging.DEBUG)


torch.set_printoptions(profile="full")

def load_input_tensors_for_npu_reshape_and_cache_bnsd(load_path: str) -> dict[str, torch.Tensor]:
    try:
        loaded_tensors = torch.load(load_path)
        hashk_op = loaded_tensors['hashk_op'].npu()
        hashk_cache_op = loaded_tensors['hashk_cache_op'].npu()
        slot_mapping_op = loaded_tensors['slot_mapping_op'].npu()
        seq_lens_op = loaded_tensors['seq_lens_op'].npu()

        input_tensors_for_reshape_and_cache_bnsd = {
            'hashk_op': hashk_op,
            'hashk_cache_op': hashk_cache_op,
            'slot_mapping_op': slot_mapping_op,
            'seq_lens_op': seq_lens_op
        }

        return input_tensors_for_reshape_and_cache_bnsd

    except FileNotFoundError:
        print(f"Error: The file '{load_path}' was not found. Please ensure the saving code was run first.")
    except Exception as e:
        print(f"An error occurred during loading: {e}")

class TestCustomReshapeAndCacheBnsd(TestCase):
    def test_reshape_and_cache_bnsd(self):
        load_path = "tests/kvcomp/npu_reshape_and_cache_bnsd_1009155843_rank_0_layerid_0.pt"
        input_tensors = load_input_tensors_for_npu_reshape_and_cache_bnsd(load_path)

        hashk_op = input_tensors['hashk_op']
        hashk_cache_op = input_tensors['hashk_cache_op']
        slot_mapping_op = input_tensors['slot_mapping_op']
        seq_lens_op = input_tensors['seq_lens_op']
        print(f'hashk_op.shape={hashk_op.shape}')
        print(f'hashk_cache_op.shape={hashk_cache_op.shape}')
        print(f'slot_mapping_op.shape={slot_mapping_op.shape}')
        print(f'seq_lens_op.shape={seq_lens_op.shape}')
        print(f'seq_lens_op={seq_lens_op}')

        num_blocks = hashk_cache_op.shape[0]
        num_kv_heads = hashk_cache_op.shape[1]
        block_size = hashk_cache_op.shape[2]
        head_size = hashk_cache_op.shape[3]
        token_num = hashk_op.shape[0] // num_kv_heads
        bs = 1
        hashk_op_org = hashk_op.reshape(num_kv_heads, bs, head_size)

        print(hashk_op_org)

        out_put = torch.ops._C_ascend.npu_reshape_and_cache_bnsd(hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, hashk_cache_op)
        print(f'out_put = {out_put.shape}')

        # verify
        for token_id in range(token_num):
        # for token_id in range(10):
            slot = slot_mapping_op[token_id].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            print(f"Token_id={token_id}, slot={slot}, block_idx={block_idx}, block_offset={block_offset}")
            for kv_head_id in range(num_kv_heads):
                # input_hashk = hashk_op_org[token_id,kv_head_id,:]
                input_hashk = hashk_op_org[kv_head_id, token_id, :]
                print()
                output_hashk = out_put[block_idx,kv_head_id,block_offset,:]
                src_idx = kv_head_id * num_blocks * 2 + token_id * 16
                dst_idx = block_idx * 2 * 128 * 16 + kv_head_id * 128 * 16 + block_offset * 16
                print(f"src_idx={src_idx}, dst_idx={dst_idx}")
                print(f"Token_id={token_id}, KV_head_id={kv_head_id}, input_hashk=output_hashk: {torch.allclose(input_hashk,output_hashk)}")
                print(f"Input Token_id={token_id}, KV_head_id={kv_head_id}, hashk={input_hashk}")
                print(f"Output Token_id={token_id}, KV_head_id={kv_head_id} reshaped hashk_cache={output_hashk}")
                print()
        
        token_id = bs - 1
        slot = slot_mapping_op[token_id].item() + 1
        block_idx = slot // block_size
        block_offset = slot % block_size

        for kv_head_id in range(num_kv_heads):
            output_hashk = out_put[block_idx,kv_head_id,block_offset,:]
            src_idx = kv_head_id * num_blocks * 2 + (token_id + 1) * 16
            dst_idx = block_idx * 2 * 128 * 16 + kv_head_id * 128 * 16 + block_offset * 16
            print(src_idx, "->", dst_idx)
            print("slot:", slot, "block_idx", block_idx, "block_off", block_offset)
            print(f"Output Token_id={token_id + 1}, KV_head_id={kv_head_id} reshaped hashk_cache={output_hashk}")
            print()

    def test_reshape_and_cache_bnsd_graph(self):
        load_path = "tests/kvcomp/npu_reshape_and_cache_bnsd_1009155843_rank_0_layerid_0.pt"
        input_tensors = load_input_tensors_for_npu_reshape_and_cache_bnsd(load_path)

        hashk_op = input_tensors['hashk_op']
        hashk_cache_op = input_tensors['hashk_cache_op']
        slot_mapping_op = input_tensors['slot_mapping_op']
        seq_lens_op = input_tensors['seq_lens_op']
        print(f'hashk_op.shape={hashk_op.shape}')
        print(f'hashk_cache_op.shape={hashk_cache_op.shape}')
        print(f'slot_mapping_op.shape={slot_mapping_op.shape}')
        print(f'seq_lens_op.shape={seq_lens_op.shape}')
        print(f'seq_lens_op={seq_lens_op}')

        num_blocks = hashk_cache_op.shape[0]
        num_kv_heads = hashk_cache_op.shape[1]
        block_size = hashk_cache_op.shape[2]
        head_size = hashk_cache_op.shape[3]
        token_num = hashk_op.shape[0] // num_kv_heads
        bs = 1
        hashk_op_org = hashk_op.reshape(num_kv_heads, bs, head_size)

        print(hashk_op_org)

        # start run custom ops
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, k_cache_out):
                
                out1 = torch.ops._C_ascend.npu_reshape_and_cache_bnsd(hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, k_cache_out)

                return out1
            
        print(f'======================== PTA graph BEGIN ========================')
        device_id = 0
        npu_mode = Network().to("npu:%s" % device_id)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        torch.npu.set_device(device_id)
        npu = f'npu:{device_id}'

        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        npu_out = npu_mode(hashk_op, hashk_cache_op, slot_mapping_op, seq_lens_op, hashk_cache_op)

        print(f'out_put = {npu_out.shape}')

        # verify
        for token_id in range(token_num):
        # for token_id in range(10):
            slot = slot_mapping_op[token_id].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            print(f"Token_id={token_id}, slot={slot}, block_idx={block_idx}, block_offset={block_offset}")
            for kv_head_id in range(num_kv_heads):
                # input_hashk = hashk_op_org[token_id,kv_head_id,:]
                input_hashk = hashk_op_org[kv_head_id, token_id, :]
                output_hashk = npu_out[block_idx,kv_head_id,block_offset,:]
                src_idx = kv_head_id * num_blocks * 2 + token_id * 16
                dst_idx = block_idx * 2 * 128 * 16 + kv_head_id * 128 * 16 + block_offset * 16
                print(f"src_idx={src_idx}, dst_idx={dst_idx}")
                print(f"Token_id={token_id}, KV_head_id={kv_head_id}, input_hashk=output_hashk: {torch.allclose(input_hashk,output_hashk)}")
                print(f"Input Token_id={token_id}, KV_head_id={kv_head_id}, hashk={input_hashk}")
                print(f"Output Token_id={token_id}, KV_head_id={kv_head_id} reshaped hashk_cache={output_hashk}")
                print()
        
        token_id = bs - 1
        slot = slot_mapping_op[token_id].item() + 1
        block_idx = slot // block_size
        block_offset = slot % block_size

        for kv_head_id in range(num_kv_heads):
            output_hashk = npu_out[block_idx,kv_head_id,block_offset,:]
            src_idx = kv_head_id * num_blocks * 2 + (token_id + 1) * 16
            dst_idx = block_idx * 2 * 128 * 16 + kv_head_id * 128 * 16 + block_offset * 16
            print(src_idx, "->", dst_idx)
            print("slot:", slot, "block_idx", block_idx, "block_off", block_offset)
            print(f"Output Token_id={token_id + 1}, KV_head_id={kv_head_id} reshaped hashk_cache={output_hashk}")
            print()


if __name__ == "__main__":
    run_tests()