# SPDX-License-Identifier: Apache-2.0
"""Cold selection of the paired A5 optimization implementation.

Private Core adapters require their reviewed source contracts. Selection
does not import kernels, load libraries or initialize an accelerator. Remove
this profile when equivalent upstream backend hooks replace the adapters.
"""

import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

CORE_WORKER_PATH = "distributed/kv_transfer/kv_connector/v1/hisparse/worker.py"
CORE_CONNECTOR_PATH = "distributed/kv_transfer/kv_connector/v1/hisparse/connector.py"
CORE_SOURCES = {
    "v1/hisparse/runtime.py": "89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926",
    "v1/attention/backends/mla/index_group.py": "dd3dfe9941e1e932f3d7b0c19bcbc177583e7cd6f54295a658eb9be094368cf4",
    CORE_WORKER_PATH: "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229",
    "v1/hisparse/coordinator.py": "530a488c60c3411e53eab42a8fe0624454de942204da304391a4f1c295b30666",
    CORE_CONNECTOR_PATH: "78610e02ebdd622dc8272a98d75c2a75a41d8b19ce1e72ac582bdedd19d11f13",
}
MAIN_CORE_SOURCES = {
    "v1/hisparse/runtime.py": "16e33814eb5d6b44093429fbb74aaf436e8994b11b43b07d1c39851baf563759",
    "v1/attention/backends/mla/index_group.py": "0a25c4558357f568144ac17d74c9fc796b576bf5c94817c95623c36fc9b860ba",
    CORE_WORKER_PATH: "864e2c7ce4e2cff320b4f68f8dccc9691adb829f0e50fdf6a27dc9c7885040e5",
    "v1/hisparse/coordinator.py": "28a75c07cac66e74211a9ec899a26aa0447dfe58089c1d162e760687cdb4c2e9",
    CORE_CONNECTOR_PATH: "b49518f82b1976919a3f50f21430279810473d1c657629b693d121ab3b9a76e7",
}


def validate_core_sources(package: Path, profile: str = "a5_c8_62592") -> None:
    """Reject a partial/different pairing before any common module is patched."""
    sources = MAIN_CORE_SOURCES if profile == "a5_c8_main" else CORE_SOURCES
    for relative, expected in sources.items():
        try:
            digest = hashlib.sha256((package / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        except OSError as error:
            raise RuntimeError(f"A5 HiSparse optimization requires its paired Core source: {relative}") from error
        if digest != expected:
            raise RuntimeError(f"Re-review A5 HiSparse optimization for the current Core file: {relative}")


def verify_core_source(path: Path, legacy_digest: str) -> None:
    """Accept one reviewed variant, never a mixture of current-main files.

    Existing paired adapters retain their exact legacy contract. A current-main
    file additionally requires all five reviewed main files, including the
    changed Resident layout and import-reservation lifetime contracts.
    """
    actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual == legacy_digest:
        return
    relative = next((name for name, digest in CORE_SOURCES.items() if digest == legacy_digest), None)
    if relative is None or actual != MAIN_CORE_SOURCES[relative]:
        raise RuntimeError(f"Unreviewed HiSparse source: {path.name}")
    package = path
    for _ in Path(relative).parts:
        package = package.parent
    validate_core_sources(package, "a5_c8_main")


def select_profile(requested: str, package: Path | None = None) -> str:
    if requested == "baseline":
        return requested
    if requested not in ("a5_c8_62592", "a5_c8_main"):
        raise ValueError("VLLM_ASCEND_HISPARSE_PROFILE must be baseline, a5_c8_62592 or a5_c8_main")
    if package is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("A5 HiSparse optimization requires an installed paired vLLM package")
        package = Path(spec.origin).parent
    validate_core_sources(package, requested)
    return requested


@lru_cache(maxsize=1)
def optimized_profile_enabled() -> bool:
    # Configuration/source are immutable for a worker lifetime. Restart after
    # changing either; switching captured graph owners is unsupported.
    from vllm_ascend import envs

    return select_profile(envs.VLLM_ASCEND_HISPARSE_PROFILE) != "baseline"


@lru_cache(maxsize=1)
def current_main_profile_enabled() -> bool:
    from vllm_ascend import envs

    return select_profile(envs.VLLM_ASCEND_HISPARSE_PROFILE) == "a5_c8_main"


def backend_class_path() -> str:
    module = "backend_optimized" if optimized_profile_enabled() else "backend"
    return f"vllm_ascend.distributed.kv_transfer.hisparse.{module}.AscendHiSparseBackend"
