"""Normalize addresses to Tensor-relative coordinates for the pre-integration v1 baseline."""

import dataclasses
import json
from pathlib import Path


def normalize(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(normalize(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    return value


def canonical(result, worker):
    buffers = []
    for name, tensors in worker._resources.kv_caches.items():
        for component, tensor in enumerate(tensors if isinstance(tensors, (tuple, list)) else (tensors,)):
            buffers.append(
                (
                    name,
                    component,
                    tensor.data_ptr(),
                    tensor.untyped_storage().nbytes() - tensor.storage_offset() * tensor.element_size(),
                )
            )
    result = normalize(result)

    def locate(address):
        for name, component, base, size in buffers:
            if base <= address < base + size:
                return [name, component, address - base]
        raise ValueError(address)

    for call in result[-1]:
        if call[0] in ("get", "put"):
            call[2] = [[locate(address) for address in row] for row in call[2]]
    return result


def expected_result(case_name):
    return json.loads(Path(__file__).with_name("reference_results.json").read_text())[case_name]
