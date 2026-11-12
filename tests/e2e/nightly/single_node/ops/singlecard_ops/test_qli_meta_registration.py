# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real C++ dispatcher registration without allocating NPU tensors.

Run against a rebuilt plugin. A subprocess turns a native registration abort
into a pytest failure instead of terminating the entire test worker.
"""

import importlib.util
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest


def _check_native_dispatch(fake):
    import torch
    import torch_npu  # noqa: F401
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from vllm_ascend.device.device_config import is_950

    # Load the actual library's static registrations without importing the
    # plugin's unrelated Python platform initialization path.
    package = importlib.util.find_spec("vllm_ascend")
    assert package is not None and package.submodule_search_locations
    libraries = [
        library
        for directory in package.submodule_search_locations
        for library in Path(directory).glob("vllm_ascend_C*.so")
    ]
    assert len(libraries) == 1, libraries
    torch.ops.load_library(str(libraries[0]))

    ops = torch.ops._C_ascend
    for name, count in (
        ("npu_quant_lightning_indexer_v2", 26),
        ("npu_quant_lightning_indexer_v2_metadata", 23),
        ("npu_quant_lightning_indexer_v3", 30),
        ("npu_quant_lightning_indexer_local_candidates", 11),
    ):
        assert len(getattr(ops, name).default._schema.arguments) == count

    device = "cpu" if fake else "meta"
    with FakeTensorMode() if fake else nullcontext():
        query = torch.empty(3, 32, 128, device=device, dtype=torch.float8_e4m3fn)
        key = torch.empty(8, 128, 1, 128, device=device, dtype=query.dtype)
        weights = torch.empty(3, 32, device=device, dtype=torch.float32)
        key_scale = torch.empty(8, 128, 1, device=device, dtype=torch.float32)
        starts = torch.empty(2, device=device, dtype=torch.int32)
        lengths = torch.empty(1, device=device, dtype=torch.int32)
        block_table = torch.empty(1, 8, device=device, dtype=torch.int32)
        args = (query, key, weights, weights, key_scale, 512, 1)
        for dcp in (False, True):
            dcp_args = (
                dict(
                    enable_dcp=True,
                    cp_world=8,
                    cp_rank=7,
                    cp_kv_cache_interleave_size=128,
                    causal_seqlens_kv_global=lengths,
                )
                if dcp
                else {}
            )
            metadata = ops.npu_quant_lightning_indexer_v2_metadata(
                32,
                1,
                128,
                512,
                1,
                cu_seqlens_q=starts,
                seqused_k=lengths,
                batch_size=1,
                **dcp_args,
            )
            assert metadata.shape == (1024,) and metadata.dtype == torch.int32
            assert metadata.device.type == device
            for return_value in (0, 1):
                kwargs = dict(
                    cu_seqlens_q=starts,
                    seqused_k=lengths,
                    block_table=block_table,
                    metadata=metadata,
                    return_value=return_value,
                    **dcp_args,
                )
                for name in ("npu_quant_lightning_indexer_v2", "npu_quant_lightning_indexer_v3"):
                    outputs = getattr(ops, name)(*args, **kwargs)
                    indices, scores = outputs[:2]
                    assert indices.shape == (3, 1, 512) and indices.dtype == torch.int32
                    assert scores.shape == ((3, 1, 512) if return_value else (0,))
                    assert scores.dtype == (torch.bfloat16 if is_950() or not return_value else torch.float32)
                    assert all(output.device.type == device for output in outputs)
                    if fake:
                        assert all(isinstance(output, FakeTensor) for output in outputs)
                    if name.endswith("v3"):
                        assert outputs[2].shape == (0,) and outputs[2].dtype == torch.int32


@pytest.mark.parametrize("mode", ["meta", "fake"])
def test_qli_native_registration_and_dispatch(mode):
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # Do not prepend this test's directory: its triton/ fixtures would
            # shadow the installed Triton package used by FakeTensor.
            "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')",
            str(Path(__file__).resolve()),
            mode,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_native_dispatch(fake=sys.argv[1] == "fake")
