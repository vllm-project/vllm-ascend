import hashlib
import logging
import sys
import threading
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
from pathlib import Path

logger = logging.getLogger(__name__)

MOONCAKE_DISTRIBUTIONS = (
    "mooncake-transfer-engine-npu",
    "mooncake-transfer-engine-ascend",
    "mooncake-transfer-engine",
)


def _get_installed_mooncake_distributions() -> list[tuple[str, importlib_metadata.Distribution]]:
    installed_distributions = []
    for distribution_name in MOONCAKE_DISTRIBUTIONS:
        try:
            distribution = importlib_metadata.distribution(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        installed_distributions.append((distribution_name, distribution))
    return installed_distributions


def _find_mooncake_engine_file() -> str:
    engine_module = sys.modules.get("mooncake.engine")
    module_file = getattr(engine_module, "__file__", "")
    if module_file:
        return str(module_file)
    try:
        module_spec = importlib_util.find_spec("mooncake.engine")
    except (ImportError, ValueError):
        return ""
    return str(getattr(module_spec, "origin", "") or "")


def validate_mooncake_runtime_installation() -> None:
    """Reject namespace collisions and stale Mooncake extension modules."""
    installed_distributions = _get_installed_mooncake_distributions()
    if len(installed_distributions) > 1:
        names = ", ".join(name for name, _ in installed_distributions)
        raise RuntimeError(
            "Multiple Mooncake distributions are installed in the same Python environment: "
            f"{names}. Uninstall all Mooncake distributions and install only mooncake-transfer-engine-npu."
        )

    if not installed_distributions:
        return

    distribution_name, distribution = installed_distributions[0]
    recorded_engine_files: list[Path] = []
    for package_file in distribution.files or ():
        relative_path = Path(str(package_file))
        if (
            len(relative_path.parts) >= 2
            and relative_path.parts[-2] == "mooncake"
            and relative_path.name.startswith("engine")
            and relative_path.suffix == ".so"
        ):
            recorded_engine_files.append(Path(distribution.locate_file(package_file)).resolve())

    module_file = _find_mooncake_engine_file()
    if recorded_engine_files and not module_file:
        expected_files = ", ".join(str(path) for path in recorded_engine_files)
        raise RuntimeError(
            f"Mooncake engine module cannot be resolved for installed distribution {distribution_name}. "
            f"Expected one of: {expected_files}. A stale mooncake package directory is likely shadowing the "
            "installed wheel. Remove stale Mooncake packages before Python starts, then reinstall "
            "mooncake-transfer-engine-npu."
        )
    if not module_file:
        return

    loaded_engine_file = Path(module_file).resolve()
    if recorded_engine_files and loaded_engine_file not in recorded_engine_files:
        expected_files = ", ".join(str(path) for path in recorded_engine_files)
        raise RuntimeError(
            f"Loaded Mooncake engine module {loaded_engine_file} is not owned by installed distribution "
            f"{distribution_name}. Expected one of: {expected_files}. A stale engine.cpython-*.so is likely "
            "shadowing the installed wheel. Remove stale Mooncake extension files before Python starts, then "
            "reinstall mooncake-transfer-engine-npu."
        )


def get_mooncake_runtime_identity() -> tuple[str, str]:
    """Return the installed package versions and loaded engine binary digest."""
    installed_distributions = [
        f"{name}=={distribution.version}" for name, distribution in _get_installed_mooncake_distributions()
    ]

    module_file = _find_mooncake_engine_file()
    module_digest = "unavailable"
    if module_file:
        try:
            digest = hashlib.sha256()
            with Path(module_file).open("rb") as module_handle:
                while chunk := module_handle.read(1024 * 1024):
                    digest.update(chunk)
            module_digest = digest.hexdigest()
        except OSError:
            logger.warning("Unable to hash the loaded Mooncake engine module: %s", module_file)

    package_identity = ",".join(installed_distributions) or "unknown-distribution"
    return f"{package_identity};engine_sha256={module_digest}", str(module_file)


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.additional_registered_buffers: set[tuple[int, int]] = set()
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    validate_mooncake_runtime_installation()
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True

    def register_additional_buffer(self, ptrs: list[int], sizes: list[int]):
        """Register buffers allocated after the initial KV cache registration."""
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            for ptr, size in zip(ptrs, sizes, strict=True):
                region = (ptr, size)
                if region in self.additional_registered_buffers:
                    continue
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake additional memory registration failed: ptr={ptr}, size={size}, ret={ret_value}"
                    )
                self.additional_registered_buffers.add(region)


global_te = GlobalTE()


if __name__ == "__main__":
    validate_mooncake_runtime_installation()
    runtime_identity, engine_module_path = get_mooncake_runtime_identity()
    if not engine_module_path:
        raise RuntimeError("Mooncake engine module is not installed or cannot be resolved.")
    print(f"Mooncake runtime validation passed: {runtime_identity} (module={engine_module_path})")
