# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only observations of native MXFP8 receiver tensors."""

import hashlib

import torch

from tests.e2e.common.rlhf.common import post_control


class MXFP8AuditWorker:
    """Inspect representative dense/MoE layers without changing reload behavior."""

    def mxfp8_snapshot(self):
        tensors = {}
        kinds = set()
        for prefix, layer in self.model_runner.model.named_modules():
            if not hasattr(layer, "_mxfp8_original_shapes"):
                continue
            kind = "moe" if "w13_weight" in layer._mxfp8_original_shapes else "linear"
            if kind in kinds:
                continue
            assert layer._mxfp8_transformed, f"{prefix}: MXFP8 post-processing was skipped"
            kinds.add(kind)
            for name in ("w13_weight", "w2_weight") if kind == "moe" else ("weight",):
                for tensor_name in (name, name + "_scale"):
                    tensor = getattr(layer, tensor_name, None)
                    if tensor is None:
                        values = getattr(layer, tensor_name + "_list", None)
                        if values is None:
                            values = getattr(layer, "cann_mega_moe_" + tensor_name + "_list")
                        tensor = values[0]
                    elif kind == "moe":
                        tensor = tensor[0]
                    expected_dtypes = (
                        (torch.float8_e4m3fn,) if tensor_name == name else (torch.uint8, torch.float8_e8m0fnu)
                    )
                    assert tensor.dtype in expected_dtypes, (prefix, tensor_name, tensor.dtype)
                    assert tensor.element_size() == 1, (
                        prefix,
                        tensor_name,
                        tensor.dtype,
                    )
                    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    tensors[prefix + "." + tensor_name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
        assert kinds == {"linear", "moe"}, f"Missing MXFP8 layer types: {kinds}"
        return tensors


def capture_quantized_result(server, case):
    from tests.e2e.common.rlhf.weight_transfer.reference_check import (
        generation_signature,
    )

    return {
        "generation": generation_signature(server.get_client(), case.model, prompts=case.prompts),
        "tensors": post_control(server, "collective_rpc", json={"method": "mxfp8_snapshot"}).json()["results"],
    }


def assert_quantized_updates(references, updates, case):
    assert len(references) == 2 and len(updates) == 3, f"{case.id}: expected A/B/A and two fresh-load baselines"
    for index, actual in enumerate(updates):
        assert actual == references[index % 2], f"{case.id}: round {index + 1} differs from fresh-load"
    assert updates[0]["generation"] != updates[1]["generation"], "A/B inference results did not change"
    for rank_a, rank_b in zip(updates[0]["tensors"], updates[1]["tensors"], strict=True):
        assert rank_a.keys() == rank_b.keys()
        for name in rank_a:
            assert rank_a[name]["sha256"] != rank_b[name]["sha256"], f"{name}: A/B value unchanged"
