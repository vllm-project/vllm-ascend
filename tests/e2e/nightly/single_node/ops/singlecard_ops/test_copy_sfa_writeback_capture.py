# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Copy-SFA writeback stream capture with NPU destination descriptors.

Device tensors stand in for the NPU-addressable Host KV views, isolating ACL
event ordering from Mooncake allocation and tensor-parallel communication.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (  # noqa: E402
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import MooncakeHostPool  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)

MAX_ROWS = 4
HOST_SLOTS = 16
DRAFT_SLOT_OFFSET = 8

pytestmark = pytest.mark.skipif(not torch_npu.npu.is_available(), reason="requires an Ascend NPU")


def make_manager():
    result = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    result.use_fused_copy_sfa = True
    result.use_fused_overlap = False
    result._host_kv_allocator = object.__new__(MooncakeHostPool)
    result.tp_rank = 0
    result.tp_size = 1
    result.layer_name_to_offload_id = {f"layer.{index}": index for index in range(4)}
    result.current_kv_save_stream = torch_npu.npu.Stream()
    element_size = torch.bfloat16.itemsize
    result.token_size_bytes_k = 2 * element_size
    result.token_size_bytes_v = element_size
    result.max_d2h_index_copy_tokens = MAX_ROWS
    result.d2h_dst_idx_npu = torch.zeros(MAX_ROWS, dtype=torch.int64, device="npu")
    result._initialize_mooncake_index_copy_descriptors()
    return result


@pytest.mark.parametrize("warmup", [False, True])
def test_copy_sfa_writeback_capture_after_warmup_and_across_shapes(warmup):
    result = make_manager()
    physical_k = torch.zeros((HOST_SLOTS + 1, 2), dtype=torch.bfloat16, device="npu")
    physical_v = torch.zeros((HOST_SLOTS + 1, 1), dtype=torch.bfloat16, device="npu")
    host_k, host_v = physical_k[:-1], physical_v[:-1]
    result._mooncake_writeback_raw_views = {
        host_k.data_ptr(): physical_k.view(torch.int8),
        host_v.data_ptr(): physical_v.view(torch.int8),
    }
    context = SimpleNamespace(is_draft_model=False)

    def offload_pair(first_layer, slots, k, v):
        for layer in (first_layer, first_layer + 1):
            result.offload_new_kv(f"layer.{layer}", slots, host_k, host_v, None, None, k, v, capturing=True)
            result.ensure_copy_sfa_host_visibility(capturing=True)

    captures = []
    with (
        patch.object(manager_module, "is_forward_context_available", return_value=True),
        patch.object(manager_module, "get_forward_context", side_effect=lambda: context),
    ):
        if warmup:
            slots = torch.arange(MAX_ROWS, dtype=torch.int64, device="npu") + 2
            k = torch.ones((MAX_ROWS, 2), dtype=torch.bfloat16, device="npu")
            v = torch.ones((MAX_ROWS, 1), dtype=torch.bfloat16, device="npu")
            offload_pair(0, slots, k, v)
            torch_npu.npu.synchronize()
        # Keep earlier graphs alive: their side-stream events must not be
        # mistaken for records belonging to a subsequent capture.
        for token_count in (MAX_ROWS, 2, MAX_ROWS):
            slots = torch.arange(token_count, dtype=torch.int64, device="npu") + 2
            k = torch.ones((token_count, 2), dtype=torch.bfloat16, device="npu")
            v = torch.ones((token_count, 1), dtype=torch.bfloat16, device="npu")
            draft_k, draft_v = k.clone(), v.clone()
            context = SimpleNamespace(is_draft_model=False)
            graph = torch_npu.npu.NPUGraph()
            torch_npu.npu.synchronize()
            with torch_npu.npu.graph(graph):
                offload_pair(0, slots, k, v)
                context.is_draft_model = True
                slots.add_(DRAFT_SLOT_OFFSET)
                offload_pair(2, slots, draft_k, draft_v)
                slots.sub_(DRAFT_SLOT_OFFSET)
            captures.append((graph, slots, k, v, draft_k, draft_v))
        # Replay older graphs with fresh KV and slots after all captures have
        # shared the descriptor buffers. Descriptors must be rebuilt for each layout.
        for step in (1, 2):
            for graph, slots, k, v, draft_k, draft_v in reversed(captures):
                slots.copy_(torch.arange(slots.numel(), dtype=torch.int64, device="npu") + step)
                k.fill_(step * 10)
                v.fill_(step * 20)
                draft_k.fill_(step * 30)
                draft_v.fill_(step * 40)
                graph.replay()
                torch_npu.npu.synchronize()
                target_slots = slots.cpu()
                draft_slots = target_slots + DRAFT_SLOT_OFFSET
                torch.testing.assert_close(host_k.cpu()[target_slots], k.cpu())
                torch.testing.assert_close(host_v.cpu()[target_slots], v.cpu())
                torch.testing.assert_close(host_k.cpu()[draft_slots], draft_k.cpu())
                torch.testing.assert_close(host_v.cpu()[draft_slots], draft_v.cpu())
