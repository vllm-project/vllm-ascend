# SPDX-License-Identifier: Apache-2.0
# Kernel source: vllm/v1/worker/gpu/metrics/logits.py
# Coverage: _num_nans_kernel
"""
Precision test for _num_nans_kernel.

Kernel signature:
    _num_nans_kernel(
        logits_ptr,               # fp32 logits [num_reqs, vocab_size]
        logits_stride,            # stride(0) of logits
        num_nans_ptr,             # int32 output [num_reqs]
        vocab_size,               # vocab size
        BLOCK_SIZE: tl.constexpr, # block size for iteration
    )

Counts NaN values in logits per request. Uses libdevice.isnan to detect NaNs
and sums them per row. The upstream kernel has no Ascend-specific
implementation, so it is validated against a CPU reference.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip(
        "Ascend NPU is required for this accuracy test",
        allow_module_level=True,
    )

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError) as exc:
    pytest.fail(f"Triton runtime is unavailable: {exc}", pytrace=False)


def _install_vllm_triton_utils_shim() -> types.ModuleType:
    """Install a package-shaped shim for ``vllm.triton_utils``.

    triton-ascend 3.2.x predates ``triton.experimental.gluon`` (and
    ``triton.experimental.gluon.nvidia``), but ``vllm.triton_utils``
    unconditionally imports gluon, so merely importing ``vllm.triton_utils``
    raises ModuleNotFoundError on this host.

    Instead of replacing the whole package, install a *package-shaped* shim:
    set ``__path__`` to the real ``triton_utils`` directory so submodules like
    ``allocation`` / ``libdevice`` / ``importing`` still load from disk, and
    expose ``tl`` / ``triton`` from the installed triton. Must run BEFORE any
    ``vllm.*`` import.
    """
    existing = sys.modules.get("vllm.triton_utils")
    if existing is not None:
        # Already imported: either our shim or a real module on a host whose
        # Triton has gluon. Both are usable as-is.
        return existing

    # Locate the real package directory without importing it.
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        pytest.fail("Could not locate the vllm package on this host", pytrace=False)
    triton_utils_dir = Path(list(spec.submodule_search_locations)[0]) / "triton_utils"

    def _placeholder_lazy(name: str):
        def _fn(*args, **kwargs):  # pragma: no cover
            del args, kwargs
            raise RuntimeError(
                f"vLLM feature '{name}' (triton.experimental.gluon) is "
                "unavailable under the shim for triton-ascend 3.2.x"
            )

        _fn.__name__ = name
        return _fn

    shim = types.ModuleType("vllm.triton_utils")
    shim.__package__ = "vllm.triton_utils"
    shim.__path__ = [str(triton_utils_dir)]
    shim.HAS_TRITON = True
    shim.triton = triton
    shim.tl = tl
    shim.tldevice = None
    try:
        import triton.language.extra.libdevice as _tldevice
        shim.tldevice = _tldevice
    except ImportError:
        pass
    shim.LOG2E = 1.4426950408889634
    shim.LOGE2 = 0.6931471805599453
    shim.gluon = _placeholder_lazy("gluon")
    shim.gl = _placeholder_lazy("gl")
    shim.aggregate = None
    try:
        from triton.language.core import _aggregate as _aggregate_ref
        shim.aggregate = _aggregate_ref
    except ImportError:
        pass
    shim.use_tensor_descriptor = _placeholder_lazy("use_tensor_descriptor")
    sys.modules["vllm.triton_utils"] = shim
    return shim


# Must execute BEFORE any vllm.* import.
_install_vllm_triton_utils_shim()

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.worker.gpu.metrics import logits as _metrics_logits  # noqa: E402

# The upstream kernel imports its libdevice from
# ``torch._inductor.runtime.triton_helpers``, which on Ascend resolves
# ``libdevice.isnan`` to an unsupported CUDA symbol that returns None at
# compile time (``AttributeError: 'NoneType' object has no attribute 'to'``).
# Rebind the module-level libdevice to the CANN libdevice so ``isnan`` resolves
# to a backend-supported symbol before the kernel is compiled (mirrors
# ``vllm_ascend/patch/worker/patch_v2/patch_triton.py``).
try:
    _metrics_logits.libdevice = triton.language.extra.cann.libdevice
except Exception as exc:  # noqa: BLE001
    pytest.skip(
        "triton.language.extra.cann.libdevice is unavailable on this host; "
        f"_num_nans_kernel cannot compile: {exc}",
        allow_module_level=True,
    )

from vllm.v1.worker.gpu.metrics.logits import _num_nans_kernel  # noqa: E402

_NUM_AICORE = -1
_NUM_VECTORCORE = -1


def _init_device_properties_triton() -> None:
    """Initialize the Ascend Triton driver's device properties (core counts).

    Inlined instead of importing ``vllm_ascend`` so this test does not pull in
    the heavyweight ``vllm_ascend.ops`` package.
    """
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE > 0 and _NUM_VECTORCORE > 0:
        return

    properties: dict = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    _NUM_AICORE = int(properties.get("num_aicore", -1))
    _NUM_VECTORCORE = int(properties.get("num_vectorcore", -1))
    if _NUM_AICORE <= 0 or _NUM_VECTORCORE <= 0:
        raise RuntimeError(f"Failed to detect Ascend Triton device properties: {properties}")


def _num_nans_ref(logits: torch.Tensor) -> torch.Tensor:
    """CPU reference: count NaNs row-wise."""
    num_reqs, vocab_size = logits.shape
    out = torch.empty(num_reqs, dtype=torch.int32)
    for i in range(num_reqs):
        count = 0
        for j in range(vocab_size):
            if torch.isnan(logits[i, j]):
                count += 1
        out[i] = count
    return out


class TestNumNansKernel:

    @pytest.fixture(autouse=True)
    def setup(self):
        _init_device_properties_triton()
        self.device = torch.device("npu")

    @pytest.mark.parametrize("num_reqs", [1, 2, 4, 8])
    @pytest.mark.parametrize("vocab_size", [128, 1024, 8192, 16384])
    @pytest.mark.parametrize("frac_nan", [0.0, 0.1, 0.5, 1.0])
    def test_num_nans(self, num_reqs, vocab_size, frac_nan):
        """Compare kernel NaN count with the CPU reference."""
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=self.device)
        # Inject NaNs at the requested fraction.
        num_nan = int(vocab_size * frac_nan)
        if num_nan > 0:
            for i in range(num_reqs):
                logits[i, :num_nan] = float("nan")

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = _num_nans_ref(logits.cpu())
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)

    def test_no_nans(self):
        """When there are no NaNs, all counts should be zero."""
        num_reqs, vocab_size = 4, 4096
        logits = torch.ones(num_reqs, vocab_size, dtype=torch.float32, device=self.device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = torch.zeros(num_reqs, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)

    def test_all_nans(self):
        """When all values are NaN, each request should report vocab_size NaN."""
        num_reqs, vocab_size = 3, 512
        logits = torch.full((num_reqs, vocab_size), float("nan"), dtype=torch.float32, device=self.device)

        num_nans = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        _num_nans_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            num_nans,
            vocab_size,
            BLOCK_SIZE=8192,
        )
        torch.npu.synchronize()

        expected = torch.full((num_reqs,), vocab_size, dtype=torch.int32)
        torch.testing.assert_close(num_nans.cpu(), expected, rtol=0, atol=0)
