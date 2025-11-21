# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

from omni.adaptors.vllm.patches.pangu_patch import patch_pangu
from omni.adaptors.vllm.patches.shm_bug_fix_patch import patch_shm_to_zmq


def patch_vllm_distributed():
    from omni.adaptors.vllm.distributed.parallel_state import (
        GroupCoordinator, initialize_model_parallel)
    from vllm import distributed

    distributed.parallel_state.GroupCoordinator = GroupCoordinator
    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
    print("++++++++++++++++++++++++patch_vllm_distributed++++++++++++++++++++++++++")
 
def patch_rope():
    from omni.models.common.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers import rotary_embedding
    rotary_embedding.get_rope = get_rope
    print("+++++++++++++++++++++++patch_rope+++++++++++++++++++++++++++")
 
def patch_embedding():
    from omni.models.common.layers.vocab_parallel_embedding import (
        ParallelLMHead, VocabParallelEmbedding)
    from vllm.model_executor.layers import vocab_parallel_embedding
    vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
    vocab_parallel_embedding.ParallelLMHead = ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding.forward = VocabParallelEmbedding.forward_vocab

def patch_sampler():
    from omni.models.common.layers.sampler import AscendSampler
    from vllm.model_executor.layers import sampler
    sampler.Sampler = AscendSampler
    from omni.models.common.layers.sampler import (RejectionSampler,
                                                   _multinomial)
    from vllm.model_executor.layers import rejection_sampler
    rejection_sampler.RejectionSampler = RejectionSampler
    rejection_sampler._multinomial = _multinomial
    print("++++++++++++++++++++++patch_sampler++++++++++++++++++++++++++++")

def patch_compilation():
    from omni.adaptors.vllm.compilation.decorators import \
        _support_torch_compile
    from vllm.compilation import decorators
    decorators._support_torch_compile = _support_torch_compile
    print("+++++++++++++++++++++++patch_compilation+++++++++++++++++++++++++++")

def patch_linear():
    from omni.models.common.layers.linear import AscendUnquantizedLinearMethod
    from vllm.model_executor.layers import linear
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
    patch_shm_to_zmq()
    _patch_done = True

patch_all() 
