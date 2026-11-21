# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce stale FIA lengths and check replays against eager attention."""

import torch
import torch_npu

from vllm_ascend import _experimental_fia_replay as native


def check_batch(batch: int) -> None:
    device = "npu"
    pages = 16
    block_size = 128
    q = torch.randn((batch, 16, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((batch * pages, block_size, 8 * 128), dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    table = torch.arange(batch * pages, dtype=torch.int32, device=device).view(batch, pages)
    orig_kwargs = dict(
        query=q,
        key=k,
        value=v,
        num_heads=16,
        num_key_value_heads=8,
        input_layout="TND",
        block_table=table,
        actual_seq_lengths=list(range(1, batch + 1)),
        actual_seq_lengths_kv=[1] * batch,
        block_size=block_size,
        scale=128**-0.5,
        sparse_mode=0,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**orig_kwargs)
    fia = torch_npu.npu_fused_infer_attention_score.out
    original, rebound, eager = [(torch.empty_like(q), torch.empty(1, device=device, dtype=q.dtype)) for _ in range(3)]
    fia(**orig_kwargs, workspace=workspace, out=eager)
    stale_graph, graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(stale_graph, stream=stream):
        fia(**orig_kwargs, workspace=workspace, out=original)
    with torch.npu.graph(graph, stream=stream):
        fia(**orig_kwargs, workspace=workspace, out=rebound)
        model = native.capture_model(stream.npu_stream)
    torch.npu.synchronize()

    lengths = torch.empty(batch, dtype=torch.int64, device=device)
    storage = torch.empty(native.ARGS_CAPACITY, dtype=torch.uint8, device=device)
    assert native.bind_lengths(
        model,
        [(q.data_ptr(), rebound[0].data_ptr())],
        table.data_ptr(),
        [1] * batch,
        lengths.data_ptr(),
        storage.data_ptr(),
    )
    max_abs = 0.0
    for i, base in enumerate((1, 2, 127, 128, 129, 512, 513, 1024, 2048)):
        live_lens = [max(1, base - request) for request in range(batch)]
        if i % 2 and batch >= 2:
            live_lens[-1] = 0
        table.copy_(table.roll(1, dims=1))
        lengths.copy_(torch.tensor(live_lens, dtype=torch.int64))
        graph.replay()
        fia(**(orig_kwargs | {"actual_seq_lengths_kv": live_lens}), workspace=workspace, out=eager)
        max_abs = max(max_abs, (rebound[0].float() - eager[0].float()).abs().max().item())
        torch.testing.assert_close(
            rebound[0], eager[0], rtol=1e-2, atol=4e-3, msg=f"batch={batch}, lengths={live_lens}"
        )
        if base == 129:
            stale_graph.replay()
            assert not torch.equal(original[0], eager[0])
    torch.npu.synchronize()
    print(f"Batch {batch} reproduced stale lengths. Rebound FIA max absolute error {max_abs:.8f}.", flush=True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    torch.manual_seed(522)
    for batch in (1, 2, 4, 8):
        check_batch(batch)
