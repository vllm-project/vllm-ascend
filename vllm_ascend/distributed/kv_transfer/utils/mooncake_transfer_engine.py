import logging
import subprocess
import threading

from vllm_ascend import envs

logger = logging.getLogger(__name__)


def _verify_heterogeneous_mooncake() -> None:
    """Fail-fast: confirm the loaded mooncake .so is a heterogeneous build.

    A non-heterogeneous Mooncake (built with ``USE_ASCEND`` instead of
    ``USE_ASCEND_HETEROGENEOUS``) loads cleanly but segfaults on NPU HBM
    operations (e.g. ``batch_register_memory`` -> ``updateLocalSegmentDesc``
    dereferences an uninitialized SegmentDesc). Catch this at init time with a
    clear error rather than letting the worker crash with a bare SIGSEGV.

    The check inspects the *actually imported* ``mooncake.engine`` module path
    (not a guessed path) for ``HeterogeneousRdmaTransport`` symbols via
    ``nm -D``. Set ``VLLM_ASCEND_MOONCAKE_HETERO_VERIFY=0`` to skip (e.g. when
    ``nm`` is unavailable, or on non-NPU builds where the guard is moot).
    """
    if not envs.VLLM_ASCEND_MOONCAKE_HETERO_VERIFY:
        return
    try:
        import mooncake.engine as me
    except ImportError:
        # Let the caller's ImportError handler produce the install hint.
        return
    so_path = getattr(me, "__file__", None)
    if not so_path:
        return
    try:
        result = subprocess.run(
            ["bash", "-c", f"nm -D '{so_path}' 2>/dev/null | grep -c HeterogeneousRdma"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        symbol_count = int(result.stdout.strip() or "0")
    except (subprocess.SubprocessError, ValueError) as e:
        logger.warning(
            "Could not verify Mooncake heterogeneous build (%s); skipping check. "
            "If NPU HBM operations segfault, ensure Mooncake is built with "
            "USE_ASCEND_HETEROGENEOUS=ON.",
            e,
        )
        return
    if symbol_count == 0:
        raise RuntimeError(
            f"Mooncake at {so_path} is not a heterogeneous build "
            f"(0 HeterogeneousRdmaTransport symbols). On NPU this causes "
            f"segfaults in KV transfer. Rebuild Mooncake with "
            f"-DUSE_ASCEND_HETEROGENEOUS=ON, or set "
            f"VLLM_ASCEND_MOONCAKE_HETERO_VERIFY=0 to skip this check."
        )


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None, heterogeneous: bool = False):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    # The heterogeneous build check only matters for MooncakeHeterogeneousConnector
                    # (910B NPU -> H20 GPU; protocol="ascend" maps to HeterogeneousRdmaTransport).
                    # protocol="ascend" maps to HeterogeneousRdmaTransport). Same-vendor NPU connectors
                    # (Layerwise/Hybrid/MooncakeConnectorV1/Store) use the old ascend transport from a USE_ASCEND
                    # build, which lacks the HeterogeneousRdmaTransport symbol; a false check would fail their
                    if heterogeneous:
                        _verify_heterogeneous_mooncake()
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


global_te = GlobalTE()
