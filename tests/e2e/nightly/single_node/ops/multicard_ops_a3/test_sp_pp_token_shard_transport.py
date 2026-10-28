# SPDX-License-Identifier: Apache-2.0
"""Exercise production PP boundaries over two stages of real processes.

Run explicitly, after reserving four idle NPUs::

    torchrun --standalone --nproc-per-node=4 <this_file> --device npu

For PP2TP8, reserve sixteen idle NPUs and add ``--tp-size 8``::

    torchrun --standalone --nproc-per-node=16 <this_file> --device npu --tp-size 8

``--device cpu`` provides a Gloo control-flow check, not NPU validation.
The test AST-loads production forwards, buffer views and GroupCoordinator's
P2P methods to avoid model weights and unrelated optional model dependencies.
Decoder bodies are deterministic test doubles; attention/kernel numerics and
ACL graph capture are outside this test's scope.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import pickle
from collections import deque, namedtuple
from datetime import timedelta
from enum import Enum
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

DEFAULT_TP_SIZE = 2
PP_SIZE = 2
HIDDEN_SIZE = 4
HC_STREAMS = 2
TOPK_WIDTH = 2
TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])
AUX_KEY = "pp_transport_aux_hidden_states_0"
TOPK_KEY = "pp_transport_topk_indices_0"
MODEL_SOURCES = (
    ("dsv4", "deepseek_v4/model.py", "DeepseekV4Model"),
    ("kimi", "kimi_k3.py", "AscendKimiLinearModel"),
    ("minimax", "minimax_m3/minimax_m3.py", "MiniMaxM3Model"),
)


class IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


class ReachedReceiverLayer(Exception):
    """Stop before the final head after checking the real receiving forward."""


def _load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)

    def node_name(node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return getattr(node, "name", None)

    nodes = [copy.deepcopy(node) for node in body if node_name(node) in names]
    assert {node_name(node) for node in nodes} == set(names), (path, names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _load_production(repo, vllm_root):
    namespace = {
        "torch": torch,
        "pickle": pickle,
        "deque": deque,
        "TensorMetadata": TensorMetadata,
    }
    parallel_source = vllm_root / "vllm/distributed/parallel_state.py"
    _load_nodes(parallel_source, ("_RetainedHandle", "_split_tensor_dict"), namespace)
    methods = (
        "isend_object",
        "recv_object",
        "_should_use_all_gather",
        "_reap_completed_isends",
        "isend_tensor_dict",
        "irecv_tensor_dict",
    )
    _load_nodes(parallel_source, methods, namespace, "GroupCoordinator")
    coordinator = type("ProductionP2PCoordinator", (), {name: namespace[name] for name in methods})
    namespace.update({"IntermediateTensors": IntermediateTensors, "Enum": Enum, "_PP_TRANSPORT_PREFIX": "pp_transport"})
    _load_nodes(
        repo / "vllm_ascend/worker/v2/pp_transport.py",
        (
            "zero_pp_shard_padding",
            "PPTokenShardBuffer",
            "PPTransportDataType",
            "_get_transport_key_prefix",
            "get_pp_transport_tensors",
            "add_pp_transport_tensors",
        ),
        namespace,
    )
    return coordinator, namespace


def _make_groups(rank, backend, coordinator_type, tp_size):
    cpu_pairs = {}
    device_pairs = {}
    tp_groups = {}
    for lane in range(tp_size):
        ranks = [lane, lane + tp_size]
        cpu_pairs[lane] = dist.new_group(ranks, backend="gloo")
        device_pairs[lane] = dist.new_group(ranks, backend=backend) if backend != "gloo" else cpu_pairs[lane]
    for stage in range(PP_SIZE):
        ranks = list(range(stage * tp_size, (stage + 1) * tp_size))
        tp_groups[stage] = dist.new_group(ranks, backend=backend)
    lane = rank % tp_size
    coordinator = coordinator_type()
    coordinator.world_size = PP_SIZE
    coordinator.rank_in_group = rank // tp_size
    coordinator.ranks = [lane, lane + tp_size]
    coordinator.cpu_group = cpu_pairs[lane]
    coordinator.device_group = device_pairs[lane]
    coordinator.use_cpu_custom_send_recv = False
    coordinator.device_communicator = None
    return coordinator, tp_groups[rank // tp_size]


def _values(rows, lane, device):
    values = torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32, device=device).reshape(rows, HIDDEN_SIZE)
    return values + 1000 * (lane + 1)


def _model_forward(repo, namespace, model_source, pp, lane, tp_group, device, num_tokens, receiver, tp_size):
    model_name, relative_path, class_name = model_source
    namespace = dict(namespace)
    rows = (num_tokens + tp_size - 1) // tp_size

    def shard(tensor):
        padding = (-tensor.shape[0]) % tp_size
        if padding:
            tensor = torch.cat([tensor, tensor.new_zeros((padding, *tensor.shape[1:]))])
        count = tensor.shape[0] // tp_size
        return tensor[lane * count : (lane + 1) * count].clone()

    def gather(tensor):
        tensors = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(tensors, tensor.contiguous(), group=tp_group)
        return torch.cat(tensors)

    namespace.update(
        {
            "get_pp_group": lambda: pp,
            "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
            "is_forward_context_available": lambda: False,
            "sp_shard": shard,
            "sp_all_gather": gather,
            "islice": islice,
            "cdiv": lambda numerator, denominator: (numerator + denominator - 1) // denominator,
        }
    )
    _load_nodes(repo / "vllm_ascend/models" / relative_path, ("forward",), namespace, class_name)
    residual_keys = () if model_name == "dsv4" else ("residual",)
    model = SimpleNamespace(
        config=SimpleNamespace(attn_res_block_size=1),
        use_sequence_parallel_moe=True,
        use_sequence_parallel=True,
        pp_sharded_tensor_keys=("hidden_states", *residual_keys),
        hc_mult=HC_STREAMS,
        aux_hidden_state_layers=(1,),
        dspark_aux_capture_materialized=False,
        start_layer=1 if receiver else 0,
        end_layer=2 if receiver else 1,
    )

    def capture_aux(states, layer_index, hidden, residual):
        if layer_index in model.aux_hidden_state_layers:
            # Kimi raw aux is its prefix state, not the AttnRes bank.
            states.append(hidden if residual is None or model_name == "kimi" else hidden + residual)
        return states

    model._maybe_add_hidden_state = capture_aux
    expected = _values(rows, lane, device)

    def layer(*args, **kwargs):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
        residual = kwargs["residual"] if "residual" in kwargs else args[2]
        if receiver:
            assert hidden.shape[0] == rows
            valid_rows = max(0, min(rows, num_tokens - lane * rows))
            wanted_hidden = expected + (7 if model_name == "kimi" else 0)
            wanted_hidden[valid_rows:] = 0
            if model_name == "dsv4":
                torch.testing.assert_close(hidden, wanted_hidden[:, None, :].expand(-1, HC_STREAMS, -1))
            else:
                torch.testing.assert_close(hidden, wanted_hidden)
                wanted_residual = expected + 500
                wanted_residual[valid_rows:] = 0
                actual_residual = residual[:, 0, :] if model_name == "kimi" else residual
                torch.testing.assert_close(actual_residual, wanted_residual)
            if model_name == "minimax":
                assert kwargs["input_is_sequence_parallel"] is True
            raise ReachedReceiverLayer
        if model_name == "dsv4":
            return expected[:, None, :].expand(-1, HC_STREAMS, -1).contiguous(), None
        if model_name == "kimi":
            residual[:, 0, :].copy_(expected + 500)
            return expected, residual, torch.full_like(expected, 7)
        assert kwargs["input_is_sequence_parallel"] is False
        return expected, expected + 500

    layer.layer_idx = 1 if receiver else 0
    layer.use_sequence_parallel = True
    model.layers = [None, layer] if receiver else [layer]
    positions = torch.arange(num_tokens, device=device)
    return namespace["forward"], model, positions


def _run_case(
    repo, namespace, model_source, rank, device, coordinator, tp_group, num_tokens, buffers, tp_size, token_counts
):
    stage, lane = divmod(rank, tp_size)
    name = model_source[0]
    rows = (num_tokens + tp_size - 1) // tp_size
    pp = SimpleNamespace(is_first_rank=stage == 0, is_last_rank=stage == 1)
    forward, model, positions = _model_forward(
        repo, namespace, model_source, pp, lane, tp_group, device, num_tokens, stage == 1, tp_size
    )
    wire_packets = []

    def record_wire(operation, tensor, *args, **kwargs):
        if tensor.dtype in (torch.float32, torch.int32):
            wire_packets.append((tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return operation(tensor, *args, **kwargs)

    if stage == 0:
        output = forward(
            model, None, positions, None, inputs_embeds=torch.zeros(num_tokens, HIDDEN_SIZE, device=device)
        )
        payload = output.tensors
        assert payload["hidden_states"].shape[0] == rows
        assert payload[AUX_KEY].shape == (num_tokens, HIDDEN_SIZE)
        payload[TOPK_KEY] = torch.arange(num_tokens * TOPK_WIDTH, dtype=torch.int32, device=device).reshape(
            num_tokens, TOPK_WIDTH
        )
        namespace["zero_pp_shard_padding"](payload, model.pp_sharded_tensor_keys, num_tokens, tp_size, lane)
        original_send = dist.isend

        def send(tensor, *args, **kwargs):
            return record_wire(original_send, tensor, *args, **kwargs)

        with patch.object(dist, "isend", side_effect=send):
            handles = coordinator.isend_tensor_dict(payload, dst=1, all_gather_group=None)
        assert wire_packets == [
            (tuple(tensor.shape), tensor.numel() * tensor.element_size())
            for tensor in payload.values()
            if isinstance(tensor, torch.Tensor)
        ]
        for handle in handles:
            handle.wait()
        # Remove the already-drained entry; Gloo Work.wait is not idempotent.
        coordinator._pending_isends.clear()
    else:
        original_recv = dist.irecv

        def recv(tensor, *args, **kwargs):
            return record_wire(original_recv, tensor, *args, **kwargs)

        with patch.object(dist, "irecv", side_effect=recv):
            payload, handles, postprocess = coordinator.irecv_tensor_dict(src=0, all_gather_group=None)
        assert not postprocess, "A PP shard must not be reassembled by the transport"
        assert payload is not None
        for handle in handles:
            handle.wait()
        assert all(isinstance(tensor, torch.Tensor) for tensor in payload.values())
        assert wire_packets == [
            (tuple(tensor.shape), tensor.numel() * tensor.element_size()) for tensor in payload.values()
        ]
        if name not in buffers:
            capacity = max(token_counts)
            backing = {
                key: torch.empty(
                    (
                        (capacity + tp_size - 1) // tp_size if key in model.pp_sharded_tensor_keys else capacity,
                        *tensor.shape[1:],
                    ),
                    dtype=tensor.dtype,
                    device=device,
                )
                for key, tensor in payload.items()
            }
            buffers[name] = namespace["PPTokenShardBuffer"](
                IntermediateTensors(backing), model.pp_sharded_tensor_keys, tp_size
            )
        buffer = buffers[name]
        addresses = {key: tensor.data_ptr() for key, tensor in buffer.backing.items()}
        buffer.activate(num_tokens)
        for key, tensor in payload.items():
            assert buffer.active[key].shape == tensor.shape, (key, buffer.active[key].shape, tensor.shape)
            buffer.active[key].copy_(tensor)
        assert addresses == {key: tensor.data_ptr() for key, tensor in buffer.backing.items()}
        expected_topk = torch.arange(num_tokens * TOPK_WIDTH, dtype=torch.int32, device=device).reshape(
            num_tokens, TOPK_WIDTH
        )
        torch.testing.assert_close(buffer.active[TOPK_KEY], expected_topk)
        expected_aux = torch.cat([_values(rows, source_lane, device) for source_lane in range(tp_size)])[:num_tokens]
        if name == "kimi":
            expected_aux = expected_aux + 7
        elif name == "minimax":
            expected_aux = 2 * expected_aux + 500
        torch.testing.assert_close(buffer.active[AUX_KEY], expected_aux)
        try:
            forward(model, None, positions, buffer.active)
        except ReachedReceiverLayer:
            pass
        else:
            raise AssertionError("The receiving production forward did not reach its decoder")
        shard_bytes = sum(payload[key].numel() * payload[key].element_size() for key in model.pp_sharded_tensor_keys)
        full_bytes = sum(
            num_tokens * payload[key][0].numel() * payload[key].element_size() for key in model.pp_sharded_tensor_keys
        )
        if num_tokens % tp_size == 0:
            assert shard_bytes * tp_size == full_bytes
        else:
            assert shard_bytes * tp_size >= full_bytes
        print(
            json.dumps(
                {
                    "model": name,
                    "tp_size": tp_size,
                    "tokens": num_tokens,
                    "source_rank": lane,
                    "destination_rank": rank,
                    "backend": "hccl" if device.type == "npu" else "gloo",
                    "shard_rows": rows,
                    "activation_bytes": shard_bytes,
                    "full_activation_bytes_per_lane": full_bytes,
                    "aux_bytes": payload[AUX_KEY].numel() * payload[AUX_KEY].element_size(),
                    "topk_bytes": payload[TOPK_KEY].numel() * payload[TOPK_KEY].element_size(),
                    "payload_bytes_posted_to_backend": sum(size for _, size in wire_packets),
                    "result": "PASS",
                }
            ),
            flush=True,
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--token-counts", type=int, nargs="+", help="Physical token counts, in execution order")
    parser.add_argument("--vllm-root", type=Path)
    args = parser.parse_args()
    if args.tp_size < 2:
        parser.error("--tp-size must be at least 2")
    token_counts = args.token_counts or ((8, 3, 1, 8) if args.tp_size == 2 else (16, 17, 1, 16))
    if any(count <= 0 for count in token_counts):
        parser.error("--token-counts must contain positive integers")
    world_size = PP_SIZE * args.tp_size
    repo = next(path for path in Path(__file__).resolve().parents if (path / "vllm_ascend").is_dir())
    vllm_root = args.vllm_root or repo.parent / "vllm"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == world_size, f"PP2TP{args.tp_size} requires {world_size} processes"
    if args.device == "npu":
        import torch_npu  # noqa: F401 - registers HCCL and the NPU device

        if not torch.npu.is_available() or torch.npu.device_count() < world_size:
            raise RuntimeError(f"{world_size} reserved, visible NPUs are required; this is not a CPU fallback")
        torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}" if args.device == "npu" else "cpu")
    backend = "hccl" if args.device == "npu" else "gloo"
    # Metadata always uses Gloo; tensor operations use the explicitly requested backend.
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        coordinator_type, namespace = _load_production(repo, vllm_root)
        coordinator, tp_group = _make_groups(rank, backend, coordinator_type, args.tp_size)
        buffers = {}
        for model_source in MODEL_SOURCES:
            for num_tokens in token_counts:
                _run_case(
                    repo,
                    namespace,
                    model_source,
                    rank,
                    device,
                    coordinator,
                    tp_group,
                    num_tokens,
                    buffers,
                    args.tp_size,
                    token_counts,
                )
        if rank == 0:
            print(
                f"PASS PP2TP{args.tp_size} production boundaries with {backend}: "
                f"rank i -> i+{args.tp_size}, i=0..{args.tp_size - 1}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
