# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe real receiver tensors and compare A/B/A against independent loads."""

import hashlib
from contextlib import contextmanager

import torch
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import post_control, rlhf_server, wait_for_free_device_memory
from tests.e2e.common.rlhf.weight_transfer.weight_source import fixed_startup_checkpoint


def _snapshot_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach()
    # MXFP4 kernels consume a transposed NZ view. CANN cannot copy that
    # non-contiguous internal format to CPU (or clone it on-device). Undo the
    # view, copy the contiguous storage, then restore its logical CPU layout.
    if value.ndim >= 2 and not value.is_contiguous() and value.transpose(-1, -2).is_contiguous():
        value = value.transpose(-1, -2).cpu().transpose(-1, -2)
    else:
        value = value.cpu()
    return value.contiguous().view(torch.uint8).numpy().tobytes()


class OnlineMXAuditWorker:
    """Read-only test extension; no quantization or load behavior is patched."""

    def online_mx_snapshot(self):
        from vllm_ascend.quantization.online_mx import AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod

        tensors = {}
        pointers = {}
        kinds = set()
        for prefix, layer in self.model_runner.model.named_modules():
            method = getattr(layer, "quant_method", None)
            if not isinstance(method, (AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod)):
                continue
            kinds.add("moe" if isinstance(method, AscendOnlineMXMoEMethod) else "linear")
            names = ("w13_weight", "w2_weight") if isinstance(method, AscendOnlineMXMoEMethod) else ("weight",)
            for weight_name in names:
                weight = getattr(layer, weight_name)
                scale = getattr(layer, weight_name + "_scale")
                assert weight.dtype in (torch.float8_e4m3fn, torch.uint8), (prefix, weight.dtype)
                assert scale.dtype == torch.uint8, (prefix, scale.dtype)
                for name, tensor in ((weight_name, weight), (weight_name + "_scale", scale)):
                    key = prefix + "." + name
                    # CPU transfer also synchronizes the preceding receive/quantization.
                    raw = _snapshot_bytes(tensor)
                    tensors[key] = [str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
                    pointers[key] = tensor.data_ptr()
        assert kinds == {"linear", "moe"}, f"Missing online MX layers: {kinds}"
        return {"tensors": tensors, "pointers": pointers}


def snapshot(server):
    return post_control(server, "collective_rpc", json={"method": "online_mx_snapshot"}).json()["results"]


def capture_online_result(server, case):
    from tests.e2e.common.rlhf.weight_transfer.reference_check import generation_signature

    return {
        "generation": generation_signature(server.get_client(), case.model, prompts=case.prompts),
        "workers": snapshot(server),
    }


@contextmanager
def online_weight_transfer_servers(
    source, case, *, backend, tensor_parallel_size, device_index, gpu_memory_utilization, env_dict
):
    from tests.e2e.common.rlhf.weight_transfer.reference_check import live_update_serve_args, reference_serve_args

    references = []
    # No reference cache: each revision loads the exact BF16 payload into a new
    # process without initializing any transfer engine.
    for revision in (0, 1):
        source.set_update_round(revision)
        for rank in range(tensor_parallel_size):
            wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
        port = get_open_port()
        with (
            fixed_startup_checkpoint(source) as checkpoint_dir,
            rlhf_server(
                checkpoint_dir,
                vllm_serve_args=reference_serve_args(
                    case,
                    port=port,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                ),
                server_host="127.0.0.1",
                server_port=port,
                env_dict=env_dict,
            ) as server,
        ):
            references.append(capture_online_result(server, case))
    source.set_update_round(0)
    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    port = get_open_port()
    with rlhf_server(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend=backend,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        yield server, references, capture_online_result(server, case)


def assert_online_updates(dummy, references, updates, case):
    assert len(updates) == 3, f"{case.id}: expected A/B/A updates"
    for index, update in enumerate(updates):
        expected = references[index % 2]
        assert update["generation"] == expected["generation"], f"round {index}: fresh-load output mismatch"
        assert len(update["workers"]) == len(expected["workers"])
        for actual, fresh in zip(update["workers"], expected["workers"]):
            assert actual["tensors"] == fresh["tensors"], f"round {index}: fresh-load weight/scale mismatch"
    for rank in range(len(updates[0]["workers"])):
        initial = dummy["workers"][rank]
        a, b, again = [update["workers"][rank] for update in updates]
        assert a["pointers"] == b["pointers"] == again["pointers"], "Graph tensor addresses changed"
        assert initial["tensors"].keys() == a["tensors"].keys()
        for name in a["tensors"]:
            assert initial["tensors"][name] != a["tensors"][name], f"{name}: first update did not change tensor"
            assert a["tensors"][name] != b["tensors"][name], f"{name}: second update did not change tensor"
            assert a["tensors"][name] == again["tensors"][name], f"{name}: third update failed to restore A"
    assert updates[0]["generation"] != updates[1]["generation"], "A/B output is not distinguishable"
