# SPDX-License-Identifier: Apache-2.0
#
# This file is a part of the vllm-ascend project.

import pytest
import torch

# Importing the patch module applies install_patch() (guarded by
# contextlib.suppress). conftest also calls adapt_patch(), so the patch is
# already applied by the time tests run; this import keeps the dependency
# explicit and matches the convention in sibling test_patch_*.py files.
from vllm_ascend.patch.platform import patch_dual_chunk_rope  # noqa: F401
from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
    DualChunkRotaryEmbedding,
)
from vllm.platforms import current_platform

HEAD_SIZE = 32
ROTARY_DIM = 32
MAX_POS = 128
BASE = 10000.0
CHUNK_SIZE = 64
LOCAL_SIZE = 8
DTYPE = torch.float


def _accelerator_devices():
    """Devices to test, adapted from vllm's DEVICES pattern for NPU.

    vllm's usual pattern is:
        DEVICES = ([f"{DEVICE_TYPE}:{i}" for i in range(min(device_count(), 2))]
                   if (is_cuda_alike() or is_xpu()) else ["cpu"])
    but NPUPlatform.is_cuda_alike() is False and is_xpu() is False, so that
    pattern would fall back to ["cpu"] even on a real Ascend box. Probe
    torch.accelerator directly: if a real accelerator is present, enumerate up
    to 2 of its devices with a dynamic rank (device_count); otherwise fall back
    to cpu (the CPU-runner path, where torch.npu is mocked and .to(npu) would
    fail). This tests npu:0/npu:1 on a multi-NPU box, npu:0 on a single-NPU box,
    and cpu:0 on a CPU-only runner.
    """
    device_type = current_platform.device_type
    try:
        if torch.accelerator.is_available():
            n = torch.accelerator.device_count()
            if n > 0:
                return [(device_type, i) for i in range(min(n, 2))]
    except Exception:
        pass
    return [("cpu", 0)]


DEVICES = _accelerator_devices()


def _make_bare_embedding():
    """Build a DualChunkRotaryEmbedding without running __init__ (avoids
    CustomOp/Platform side effects) but with every attribute the real
    `_compute_cos_sin_cache` reads."""
    emb = DualChunkRotaryEmbedding.__new__(DualChunkRotaryEmbedding)
    emb.head_size = HEAD_SIZE
    emb.rotary_dim = ROTARY_DIM
    emb.max_position_embeddings = MAX_POS
    emb.base = BASE
    emb.is_neox_style = True
    emb.chunk_size = CHUNK_SIZE
    emb.local_size = LOCAL_SIZE
    emb.dtype = DTYPE
    return emb


def test_patch_is_installed():
    """Importing patch_dual_chunk_rope must replace _compute_cos_sin_cache."""
    assert (
        DualChunkRotaryEmbedding._compute_cos_sin_cache.__name__
        == "_patched_compute_cos_sin_cache"
    )


@pytest.mark.parametrize("dev_type,dev_index", DEVICES)
def test_compute_cos_sin_cache_uses_platform_device(monkeypatch, dev_type, dev_index):
    """The wrapper must set self.device to (current_platform.device_type,
    accelerator index) instead of the upstream hard-coded "cuda:idx", then run
    the real original cos/sin build end-to-end on that device.

    dev_index is dynamic - drawn from torch.accelerator.device_count() - so this
    exercises the actual card ranks present (npu:0/npu:1 on a 2-NPU box) rather
    than a hard-coded index. On the CPU runner (torch.npu mocked) it falls back
    to cpu:0.
    """
    # The closure reads current_platform.device_type dynamically, so an instance
    # attribute override is enough to steer it without touching the closure.
    monkeypatch.setattr(current_platform, "device_type", dev_type)
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda di=dev_index: di
    )

    emb = _make_bare_embedding()
    caches = DualChunkRotaryEmbedding._compute_cos_sin_cache(emb)

    # Device type follows the active platform, never the upstream "cuda".
    assert emb.device == torch.device(dev_type, dev_index)
    assert emb.device.type != "cuda"
    # The original cos/sin build ran end-to-end with the corrected device.
    chunk_len = CHUNK_SIZE - LOCAL_SIZE
    assert len(caches) == 5
    assert caches[0].shape == (chunk_len, ROTARY_DIM)  # q_cache
    assert caches[2].shape == (MAX_POS, ROTARY_DIM)  # k_cache
    assert caches[0].device.type == dev_type


@pytest.mark.parametrize("dev_type,dev_index", DEVICES)
def test_compute_cos_sin_cache_overrides_hardcoded_cuda(
    monkeypatch, dev_type, dev_index
):
    """Regression guard: even if upstream __init__ already set self.device to
    the buggy "cuda:idx", the wrapper must overwrite it with the platform
    device before the original build runs."""
    monkeypatch.setattr(current_platform, "device_type", dev_type)
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda di=dev_index: di
    )

    emb = _make_bare_embedding()
    emb.device = torch.device("cuda:0")  # simulate the buggy upstream init

    DualChunkRotaryEmbedding._compute_cos_sin_cache(emb)

    assert emb.device.type == dev_type
    assert emb.device.type != "cuda"
