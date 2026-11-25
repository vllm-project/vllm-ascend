# SPDX-License-Identifier: Apache-2.0
import atexit
import ctypes
import functools
import pathlib
import shutil
import subprocess
import tempfile
import threading
from dataclasses import make_dataclass
from types import SimpleNamespace as ns
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend import utils as ascend_utils
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import attention_fence
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise import selected_kv_plan as kv_cfg
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta

HI_GVA = (1 << 60) + 1
make_shared = make_dataclass("FakeShared", ["block_ids_arr", "block_gvas_arr", ("block_keys", object, None)])
_TF: list[Any] = ["layer_idx_in_group", "shared_block_data", "block_ranges", ("group_id", int, 0)]
make_task = make_dataclass("FakeTask", [*_TF, ("use_key_major_ranges", bool, False)])


def raises(exc, pat, fn, *a, **kw):
    with pytest.raises(exc, match=pat):
        fn(*a, **kw)


class FakeEvent:
    def __init__(self, ready: bool = True) -> None:
        self.ready, self.recorded, self.synchronized = ready, 0, 0

    def record(self, *_a, **_kw) -> None:
        self.recorded += 1

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronized += 1
        self.ready = True


def make_req(req_id, h, q, pages, gvas, *, skip_tokens=None, partial=None) -> ReqMeta:
    sp = LoadSpec(0, h, True, kvpool_store_skip_tokens=h if skip_tokens is None else skip_tokens)
    kw = dict(block_ids_by_group=[pages], block_ids_by_group_np=[np.array(pages, np.int64)])
    kw |= dict(load_block_gvas_by_group_np=[np.array(gvas, np.int64)], partial_load_gva_per_group=partial)
    return ReqMeta(req_id=req_id, target_token_len=h + q, load_spec=sp, **kw)


def make_worker_stub(num_blocks: int, layers: int, load_shared, save_shared):
    lt, st = ([[make_task(i, sh, ["r"])] for i in range(layers)] for sh in (load_shared, save_shared))
    lk, w = threading.RLock(), ns(num_blocks=num_blocks, next_layer_to_submit=0, _invalid_block_ids=set())
    w._invalid_block_ids_lock, w.layer_load_tasks, w.layer_save_tasks = lk, lt, st
    w._as_cache_tuple = lambda v: v if isinstance(v, tuple) else (v,)
    return w


@pytest.fixture
def patched_npu_env(monkeypatch):
    comp, ids, sync_m = ns(npu_stream=101, wait_event=MagicMock()), iter(range(201, 999)), MagicMock()
    cur, st, e = [comp], (lambda **_kw: ns(npu_stream=next(ids), wait_event=MagicMock())), torch.empty

    class _Sw:
        def __init__(self, x):
            self.x = x

        def __enter__(self):
            cur.append(cur[0])
            cur[0] = self.x or comp
            return cur[0]

        def __exit__(self, *_a):
            cur[0] = cur.pop()

    mp = monkeypatch.setattr
    for k, v in (("Stream", st), ("Event", FakeEvent), ("current_stream", lambda: cur[0]), ("stream", _Sw)):
        mp(torch.npu, k, v, raising=False)
    mp(torch.npu, "synchronize", sync_m, raising=False)
    mp(ascend_utils, "enable_custom_op", lambda: None, raising=False)
    mp(attention_fence, "_attention_load_gate", None)
    mp(attention_fence, "_attention_save_gate", None)
    mp(torch, "empty", lambda s, **kw: e(s, **{k: v for k, v in kw.items() if k != "pin_memory"}))
    return ns(compute_stream=comp, sync=sync_m)


def make_vllm_config(**overrides):
    ex: dict[str, Any] = dict(backend="memcache", use_layerwise=True, load_async=False, layerwise_num_shared_buffers=2)
    ex["layerwise_independent_layers"] = []
    ktc = ns(kv_role="kv_producer", kv_connector="AscendStoreConnector", kv_connector_extra_config=ex)
    mc = ns(
        enforce_eager=True,
        is_encoder_decoder=False,
        enable_sleep_mode=False,
        max_model_len=200000,
        hf_text_config=ns(model_type="glm_moe_dsa"),
        get_num_layers=lambda _pc=None: 4,
        get_total_num_hidden_layers=lambda: 4,
    )
    sc = ns(async_scheduling=False, max_num_batched_tokens=kv_cfg.MAX_BATCHED_TOKENS, max_num_seqs=kv_cfg.MAX_REQUESTS)
    pc = ns(enable_dbo=False, pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1)
    cfg = ns(
        kv_transfer_config=ktc,
        additional_config={},
        speculative_config=None,
        model_config=mc,
        attention_config=ns(hisparse_config=None, indexer_kv_dtype="int8"),
    )
    cfg.cache_config = ns(kv_cache_memory_bytes=None, num_gpu_blocks_override=None, cache_dtype="int8")
    cfg.scheduler_config, cfg.parallel_config = sc, pc
    for k, v in overrides.items():
        s, _, f = k.partition(".")
        setattr(getattr(cfg, s) if f else cfg, f or s, v)
    return cfg


def make_specs_and_cfg(
    layers: int = 4,
    idx_layers: Any = None,
    num_blocks: int = 8,
    m_dtype: Any = None,
    l_dtype: Any = None,
    s_dtype: Any = None,
):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

    from vllm_ascend.core import kv_cache_interface as kci

    idx = range(layers) if idx_layers is None else idx_layers
    ms = kci.AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=656, dtype=m_dtype or torch.int8, cache_sparse_sfa_c8=True
    )
    ls = kci.AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=l_dtype or torch.int8,
        scale_dim=1,
        scale_dtype=s_dtype or torch.float16,
        cache_sparse_li_c8=True,
    )
    specs: dict[str, Any] = {f"model.layers.{i}.self_attn.attn": ms for i in range(layers)}
    specs |= {f"model.layers.{i}.self_attn.indexer.k_cache": ls for i in idx}
    tensors = [
        KVCacheTensor(size=num_blocks * s.page_size_bytes, layers=[k], layer_stride=0, block_stride=s.page_size_bytes)
        for k, s in specs.items()
    ]
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=kv_cfg.PAGE_ROWS, kv_cache_specs=specs))
    return ms, ls, specs, KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=[group])


@functools.lru_cache(maxsize=1)
def load_host_sim():
    cc = shutil.which("c++") or pytest.skip("A host C++ compiler is required")
    hd, P = pathlib.Path(__file__).resolve().parent, pathlib.Path
    csrc, tmp = hd.parents[4] / "csrc", P(tempfile.mkdtemp(prefix="skv_"))
    atexit.register(shutil.rmtree, tmp, True)
    t_inc = [f for p in torch.utils.cpp_extension.include_paths() for f in ("-isystem", p)]
    srcs = (csrc / "selected_kv_ops.cpp", csrc / "kernels/selected_kv_kernels.cpp", hd / "kernel_operator.h")
    fl = ["-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-DSELECTED_KV_HOST_SIM", f"-I{hd}", f"-I{csrc / 'kernels'}"]
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"
    run = lambda cmd: subprocess.run(cmd, capture_output=True, text=True)
    assert (p := run([cc, "-std=c++17", *fl, "-c", str(srcs[1]), "-o", str(tmp / "k.o")])).returncode == 0, p.stderr
    for s in ("c++17", "c++20"):
        if (op := run([cc, f"-std={s}", *fl, *t_inc, abi, "-c", str(srcs[0]), "-o", str(tmp / "o.o")])).returncode == 0:
            break
    assert op.returncode == 0, op.stderr
    ld, so = P(torch.__file__).parent / "lib", tmp / "libselected_kv_host_sim.so"
    lnk = [cc, "-shared", str(tmp / "k.o"), str(tmp / "o.o"), f"-L{ld}", f"-Wl,-rpath,{ld}", "-ltorch_cpu"]
    assert (lp := run([*lnk, "-ltorch", "-lc10", "-o", str(so)])).returncode == 0, lp.stderr
    torch.ops.load_library(str(so))
    lib: Any = ctypes.CDLL(str(so))
    lib.selected_kv_sim_peak_queue.restype = lib.selected_kv_sim_trap_count.restype = ctypes.c_size_t
    lib.selected_kv_sim_set_mapper.argtypes = [ctypes.c_void_p]
    lib.selected_kv_sim_set_mapper.restype = None
    orig_reset = lib.selected_kv_sim_reset

    def _reset() -> None:
        lib.selected_kv_sim_set_mapper(None)
        orig_reset()

    lib.selected_kv_sim_reset = _reset
    return torch.ops._C_ascend_host_sim, lib
