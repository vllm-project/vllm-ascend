# SPDX-License-Identifier: Apache-2.0
"""Shared CPU & AscendC Host-Sim fixtures for HiSparse prefill demo unit tests."""

from __future__ import annotations

import atexit
import ctypes
import functools
import pathlib
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch


@functools.lru_cache(maxsize=1)
def load_prefill_host_sim():
    """Compile hisparse_prefill_kernels.cpp + hisparse_prefill_ops.cpp with -DHISPARSE_PREFILL_HOST_SIM."""
    cc = shutil.which("c++")
    if cc is None:
        pytest.skip("A host C++ compiler is required for HISPARSE_PREFILL_HOST_SIM tests")

    here = pathlib.Path(__file__).resolve().parent
    csrc = here.parents[4] / "csrc"
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="hsp_sim_"))
    atexit.register(shutil.rmtree, tmp, True)

    torch_incs = [f for p in torch.utils.cpp_extension.include_paths() for f in ("-isystem", p)]
    flags = [
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-DHISPARSE_PREFILL_HOST_SIM",
        f"-I{here}",
        f"-I{csrc / 'kernels'}",
    ]
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True)

    k_obj = tmp / "k.o"
    k_res = run([cc, "-std=c++17", *flags, "-c", str(csrc / "kernels/hisparse_prefill_kernels.cpp"), "-o", str(k_obj)])
    assert k_res.returncode == 0, k_res.stderr

    o_obj = tmp / "o.o"
    o_res = None
    for std in ("c++17", "c++20"):
        o_res = run(
            [
                cc,
                f"-std={std}",
                *flags,
                *torch_incs,
                abi,
                "-c",
                str(csrc / "hisparse_prefill_ops.cpp"),
                "-o",
                str(o_obj),
            ]
        )
        if o_res.returncode == 0:
            break
    assert o_res is not None and o_res.returncode == 0, o_res.stderr

    torch_lib = pathlib.Path(torch.__file__).parent / "lib"
    so = tmp / "libhisparse_prefill_sim.so"
    lnk = run(
        [
            cc,
            "-shared",
            str(k_obj),
            str(o_obj),
            f"-L{torch_lib}",
            f"-Wl,-rpath,{torch_lib}",
            "-ltorch_cpu",
            "-ltorch",
            "-lc10",
            "-o",
            str(so),
        ]
    )
    assert lnk.returncode == 0, lnk.stderr

    torch.ops.load_library(str(so))
    lib = ctypes.CDLL(str(so))
    lib.selected_kv_sim_peak_queue.restype = ctypes.c_size_t
    lib.selected_kv_sim_trap_count.restype = ctypes.c_size_t
    return torch.ops._C_ascend_prefill_sim, lib


@pytest.fixture(autouse=True)
def _reset_host_sim_state():
    if load_prefill_host_sim.cache_info().currsize > 0:
        load_prefill_host_sim()[1].selected_kv_sim_reset()
    yield
    if load_prefill_host_sim.cache_info().currsize > 0:
        load_prefill_host_sim()[1].selected_kv_sim_reset()


@pytest.fixture
def patched_npu_env(monkeypatch):
    """Provide CPU-safe torch.npu streams, events, and current_stream context manager."""
    stream_seq = iter(range(100, 999))
    recorded_events: list[object] = []

    class FakeEvent:
        def __init__(self, **_kwargs):
            self.recorded_on: list[int] = []
            recorded_events.append(self)

        def record(self, stream=None):
            self.recorded_on.append(getattr(stream, "npu_stream", -1))

        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            return None

    class FakeStream:
        def __init__(self, *, npu_stream: int | None = None, **_kwargs):
            self.npu_stream = next(stream_seq) if npu_stream is None else npu_stream
            self.waited_events: list[object] = []
            self.waited_streams: list[int] = []
            self.synced = 0

        def wait_event(self, ev):
            self.waited_events.append(ev)

        def wait_stream(self, other):
            self.waited_streams.append(other.npu_stream)

        def synchronize(self):
            self.synced += 1

    compute_stream = FakeStream(npu_stream=0)
    stack = [compute_stream]

    class _StreamCtx:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            stack.append(stack[-1])
            stack[-1] = self.stream or compute_stream
            return stack[-1]

        def __exit__(self, *_a):
            stack.pop()

    monkeypatch.setattr(torch.npu, "Stream", FakeStream, raising=False)
    monkeypatch.setattr(torch.npu, "Event", FakeEvent, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stack[-1], raising=False)
    monkeypatch.setattr(torch.npu, "stream", _StreamCtx, raising=False)
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock(), raising=False)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False, raising=False)
    return NS(compute_stream=compute_stream, events=recorded_events)
