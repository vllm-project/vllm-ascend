# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from omni.adaptors.vllm.patches.pangu_patch import patch_pangu

def patch_vllm_distributed():
    from vllm import distributed
    from omni.adaptors.vllm.distributed.parallel_state import (
        initialize_model_parallel,
        GroupCoordinator
    )

    distributed.parallel_state.GroupCoordinator = GroupCoordinator
    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
    print("++++++++++++++++++++++++patch_vllm_distributed++++++++++++++++++++++++++")
 
def patch_rope():
    from vllm.model_executor.layers import rotary_embedding
 
    from omni.layers.rotary_embedding import get_rope
    rotary_embedding.get_rope = get_rope
    print("+++++++++++++++++++++++patch_rope+++++++++++++++++++++++++++")
 
def patch_embedding():
    from vllm.model_executor.layers import vocab_parallel_embedding
    from omni.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
    vocab_parallel_embedding.ParallelLMHead = ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding.forward = VocabParallelEmbedding.forward_vocab

def patch_sampler():
    from omni.layers.sampler import AscendSampler
    from vllm.model_executor.layers import sampler
    sampler.Sampler = AscendSampler
    from vllm.model_executor.layers import rejection_sampler
    from omni.layers.sampler import RejectionSampler, _multinomial
    rejection_sampler.RejectionSampler = RejectionSampler
    rejection_sampler._multinomial = _multinomial
    print("++++++++++++++++++++++patch_sampler++++++++++++++++++++++++++++")

def patch_compilation():
    from omni.adaptors.vllm.compilation.decorators import _support_torch_compile
    from vllm.compilation import decorators
    decorators._support_torch_compile = _support_torch_compile
    print("+++++++++++++++++++++++patch_compilation+++++++++++++++++++++++++++")

def patch_linear():
    from vllm.model_executor.layers import linear
    from omni.layers.linear import AscendUnquantizedLinearMethod
    linear.UnquantizedLinearMethod = AscendUnquantizedLinearMethod

_patch_done = False

def patch_all():
    global _patch_done
    if _patch_done:
        return
    patch_vllm_distributed()
    patch_rope()
    patch_embedding()
    patch_sampler()
    patch_compilation()
    patch_pangu()
    patch_linear()
    _patch_done = True

patch_all() 
