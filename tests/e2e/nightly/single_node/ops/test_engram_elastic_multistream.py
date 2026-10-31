# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production hash, projection and native events across FULL gears.

Run on one NPU with synthetic fetches, or from test_engram_fp8_offload with
real multi-rank ElasticBuffer tables. A real quantized Wkv checkpoint still
requires a model test; this smoke deliberately uses a small BF16 projection.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch


def make_hash(device, rows=None):
    from vllm_ascend.models.deepseek_v41.engram.hash_state import AscendNgramHashState

    state = AscendNgramHashState.__new__(AscendNgramHashState)
    torch.nn.Module.__init__(state)
    half = rows // 2 if rows is not None else 16
    state.pad_id, state.block_size, state.use_slot_cache = 0, 16, False
    state.register_buffer("token_map", torch.arange(32, device=device))
    state.register_buffer("primes", torch.full((2, 2, 1), half - 1, device=device))
    state.register_buffer("offsets", torch.tensor([[0, half], [0, half]], device=device))
    state.register_buffer("multipliers", torch.tensor([[3, 5, 7], [5, 7, 9]], device=device))

    def packed(ids, positions, query, history, slots, blocks, valid, dead, history_dead, **kwargs):
        mask = dead[: ids.numel()] | (torch.arange(ids.numel(), device=device) >= valid)
        return state(ids, positions, query, mask, history, history < 0, None, None), ~mask

    return SimpleNamespace(forward_packed=packed)


def make_engram(layer_id, device, table=None, reference=None):
    from vllm_ascend.models.deepseek_v41.engram.layer import AscendEngram

    class Layer(torch.nn.Module):
        precompute_kv = AscendEngram.precompute_kv
        apply_precomputed = AscendEngram.apply_precomputed

        def __init__(self):
            super().__init__()
            self.layer_id, self.layer_hash_index = layer_id, (1, 14).index(layer_id)
            self.layer_name = f"smoke.layers.{layer_id}.engram"
            self.hc_mult, self.dim, self.eps, self.rotation = 2, 4, 1e-6, None
            self.embed_tokens, self.reference = table, reference
            self.calls = 0
            width = table.width if table is not None else (reference.shape[1] if reference is not None else 1)
            self.wkv = torch.nn.Linear(2 * width, 12, bias=False, dtype=torch.bfloat16)
            self.wkv.weight.data.fill_(1 / (16 * width))
            self.q_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))
            self.k_weight = torch.nn.Parameter(torch.ones(2, 4, dtype=torch.bfloat16))

        def embeddings(self, ids):
            return self.reference[ids].flatten(1) if self.reference is not None else ids.to(torch.bfloat16) / 32

        def start_prefetch(self, ids):
            self.calls += 1
            return self.embed_tokens.start_fetch(ids) if self.embed_tokens is not None else ids.clone()

        def finish_prefetch(self, handle):
            self.calls += 1
            return (
                self.embed_tokens.finish_fetch(handle).flatten(1)
                if self.embed_tokens is not None
                else self.embeddings(handle)
            )

    return Layer().to(device)


@torch.inference_mode()
def run_smoke(tables=None, reference_tables=None):
    import torch_npu  # noqa: F401
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.models.deepseek_v41.engram import ops
    from vllm_ascend.models.deepseek_v41.engram.executor import EngramMultiStreamExecutor

    if tables is None:
        torch.npu.set_device(0)
    tables, reference_tables = tables or {}, reference_tables or {}
    rows = reference_tables[1].shape[0] if reference_tables else None
    layers = {layer: make_engram(layer, "npu", tables.get(layer)) for layer in (1, 14)}
    hash_fn = make_hash("npu", rows)
    executor = EngramMultiStreamExecutor((1, 14), hash_fn=hash_fn, engram_layers=layers)
    for layer in layers.values():
        layer.executor = executor
    context = SimpleNamespace(
        no_compile_layers={layer.layer_name: layer for layer in layers.values()},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    reference = [make_engram(layer, "cpu", reference=reference_tables.get(layer)) for layer in (1, 14)]
    ids = torch.zeros(16, dtype=torch.int64, device="npu")
    positions = torch.arange(16, device="npu") + 2
    query = torch.zeros(5, dtype=torch.int32, device="npu")
    history = torch.ones((4, 2), dtype=torch.int64, device="npu")
    slots, blocks = torch.zeros(16, device="npu"), torch.zeros((4, 2), device="npu")
    valid = torch.zeros(1, dtype=torch.int32, device="npu")
    dead = torch.zeros(16, dtype=torch.bool, device="npu")
    history_dead = torch.zeros_like(history, dtype=torch.bool)
    graphs, snapshots = {}, []

    def forward(gear):
        hashes, mask = torch.ops.vllm.engram_elastic_begin(
            ids[:gear],
            positions[:gear],
            query,
            history,
            slots,
            blocks,
            valid,
            dead,
            history_dead,
            layers[1].layer_name,
            2,
            2,
        )
        hidden = ids[:gear, None, None].to(torch.bfloat16).expand(-1, 2, 4) / 32 + 1 / 32
        kv = torch.ops.vllm.engram_elastic_project(hashes, hidden, layers[1].layer_name, 2, 4)
        hidden = torch.ops.vllm.engram_elastic_consume(hidden, *kv, mask, layers[1].layer_name) + 2 / 32
        kv = torch.ops.vllm.engram_elastic_project(hashes, hidden, layers[14].layer_name, 2, 4)
        hidden = hidden + 3 / 32
        return torch.ops.vllm.engram_elastic_consume(hidden, *kv, mask, layers[14].layer_name)

    try:
        with patch.object(ops, "get_forward_context", return_value=context):
            for gear in (8, 4, 16, 2):
                forward(gear)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                context.cudagraph_runtime_mode = CUDAGraphMode.FULL
                with torch.npu.graph(graph):
                    output = forward(gear)
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                graphs[gear] = graph, output
            for use_graph in (False, True, True, False):
                for step, gear in enumerate((8, 4, 16, 8, 2)):
                    real, value = (0, gear, gear - 1)[step % 3], len(snapshots) + 1
                    ids.fill_(value % 29)
                    history.fill_(value % 13)
                    query.fill_(real)
                    query[0].zero_()
                    valid.fill_(real)
                    dead.copy_(torch.arange(16, device="npu") >= real)
                    if use_graph:
                        graph, output = graphs[gear]
                        before = [layer.calls for layer in layers.values()]
                        graph.replay()
                        assert before == [layer.calls for layer in layers.values()]
                    else:
                        output = forward(gear)
                    snapshots.append((output.clone(), gear, real, value))
                    executor.release()
        torch.npu.synchronize()
        half = rows // 2 if rows is not None else 16
        for actual, gear, real, value in snapshots:
            # Constant token chunks make a direct independent CPU n-gram reference.
            cpu_ids = torch.full((gear,), value % 29, dtype=torch.int64)
            hashes = torch.empty((gear, 2, 2), dtype=torch.int64)
            for token in range(gear):
                for layer, multipliers in enumerate(((3, 5, 7), (5, 7, 9))):
                    rolling = 0
                    for shift, multiplier in enumerate(multipliers):
                        source = value % 29 if token >= shift else value % 13
                        # Dead rows terminate the n-gram at its first token.
                        rolling ^= (source if token < real else 0) * multiplier
                        if shift:
                            hashes[token, layer, shift - 1] = rolling % (half - 1) + (half if shift == 2 else 0)
            expected = cpu_ids[:, None, None].to(torch.bfloat16).expand(-1, 2, 4) / 32 + 1 / 32
            mask = torch.arange(gear) < real
            for slot, layer in enumerate(reference):
                expected = layer.apply_precomputed(
                    expected, *layer.precompute_kv(layer.embeddings(hashes[:, slot])), mask
                )
                if slot == 0:
                    expected = expected + 2 / 32 + 3 / 32
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)
    finally:
        executor.close()


def test_engram_elastic_multistream():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
