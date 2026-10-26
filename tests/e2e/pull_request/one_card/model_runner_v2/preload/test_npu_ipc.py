# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc
import multiprocessing

import pytest
import torch
import torch_npu

from vllm_ascend.model_loader.preload.transport import NPUTensorEntry


def _consumer(connection):
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    entry, expected, stride, offset = connection.recv()
    tensor = entry.rebuild(0)
    torch.testing.assert_close(tensor.cpu(), torch.tensor(expected, dtype=tensor.dtype))
    assert tensor.stride() == stride
    assert tensor.storage_offset() == offset
    tensor.fill_(7)
    torch.npu.synchronize()
    del tensor
    gc.collect()
    connection.send("released")
    connection.close()


def _alias_consumer(connection):
    from vllm.config.load import LoadConfig

    from vllm_ascend.model_loader.preload.loader import AscendIpcModelLoader

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    state = connection.recv()
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.empty(16, 16, dtype=torch.bfloat16, device="meta"))
    model.tied = model.weight
    model.register_buffer("view", torch.empty(8, 8, dtype=torch.bfloat16, device="meta"))
    loader = AscendIpcModelLoader(LoadConfig(load_format="ipc_cache", model_loader_extra_config={"fallback": False}))
    loader._apply_entries(model, state, 0)
    assert model.weight is model.tied
    assert model.weight.untyped_storage().data_ptr() == model.view.untyped_storage().data_ptr()
    assert model.view.stride() == (16, 1) and model.view.storage_offset() == 33
    model.view.fill_(7)
    torch.npu.synchronize()
    torch.testing.assert_close(model.weight[2:10, 1:9], torch.full_like(model.view, 7))
    del model
    gc.collect()
    connection.send("released")
    connection.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("view", ["contiguous", "transpose", "offset"])
def test_nd_ipc_views_and_repeated_consumers(dtype, view):
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    base = torch.arange(256, dtype=torch.float32).to(dtype).npu().reshape(16, 16)
    tensor = base if view == "contiguous" else base.T if view == "transpose" else base[2:10, 1:9]
    context = multiprocessing.get_context("spawn")
    for _ in range(3):
        expected = tensor.cpu().tolist()
        entry = NPUTensorEntry.from_tensor(tensor, "param")
        parent, child = context.Pipe()
        process = context.Process(target=_consumer, args=(child,))
        process.start()
        child.close()
        try:
            parent.send((entry, expected, tensor.stride(), tensor.storage_offset()))
            assert parent.poll(120), "NPU IPC consumer timed out"
            assert parent.recv() == "released"
            process.join(30)
            assert process.exitcode == 0
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(30)
        torch.npu.synchronize()
        assert bool(torch.all(tensor == 7).item())
    del tensor, base
    gc.collect()
    torch.npu.empty_cache()


def test_nz_ipc_fails_explicitly():
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    tensor = torch_npu.npu_format_cast(torch.ones(32, 32, dtype=torch.bfloat16, device="npu"), 29)
    with pytest.raises(ValueError, match="NZ IPC"):
        NPUTensorEntry.from_tensor(tensor, "param")


def test_npu_ipc_preserves_tied_parameters_and_shared_views():
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheState

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    tensor = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
    state = WeightCacheState(
        entries={
            "weight": NPUTensorEntry.from_tensor(tensor, "param"),
            "view": NPUTensorEntry.from_tensor(tensor[2:10, 1:9], "buffer"),
        },
        aliases={"tied": "weight"},
        attrs={},
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_alias_consumer, args=(child,))
    process.start()
    child.close()
    try:
        parent.send(state)
        assert parent.poll(120), "NPU IPC alias consumer timed out"
        assert parent.recv() == "released"
        process.join(30)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(30)
    torch.npu.synchronize()
    torch.testing.assert_close(tensor[2:10, 1:9], torch.full_like(tensor[2:10, 1:9], 7))
    del tensor
    gc.collect()
    torch.npu.empty_cache()
