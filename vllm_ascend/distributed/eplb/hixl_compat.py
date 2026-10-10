"""Minimal ctypes binding for the CANN HIXL engine API used by EPLB.

Newer CANN distributions install an official ``hixl`` Python package next to
``libcann_hixl.so``. Environments that ship the shared library without the
package (for example CANN 9.1.0) can still drive the HIXL EPLB communicator
through this module, which calls the exported C++ symbols of the installed
toolkit directly and therefore matches the ABI of the local CANN version.

Only the entry points used by ``AscendHixlEplbCommunicator`` are bound:
``Hixl()``, ``initialize``, ``register_mem``, ``deregister_mem``, ``connect``,
``disconnect``, ``transfer_async``, ``get_transfer_status`` and ``finalize``.

The C++ ABI is not directly callable from Python, so this module replicates
the three object layouts that cross the boundary. All of them were probed
against the GCC libstdc++ / CANN 9.1.0 aarch64 toolkit (header
``hixl/hixl_types.h`` plus a ``sizeof`` probe) and are asserted by unit tests:

- ``ge::AscendString`` holds one ``std::shared_ptr<std::string>`` (16 bytes).
  It is built through the real ``AscendString(const char *, size_t)``
  constructor exported by ``libmetadef.so`` so the library itself owns the
  reference-counted storage.
- an empty ``std::map<AscendString, AscendString>`` (48 bytes) whose tree
  header self-pointers make it a valid empty map for iteration and copying.
- a ``std::vector<TransferOpDesc>`` (24 bytes) as the ``begin/end/capacity``
  pointer triple around a ctypes array.
"""

from __future__ import annotations

import ctypes
import glob
import os
import struct
import threading
from collections.abc import Callable
from enum import IntEnum
from typing import Any

from vllm.logger import logger

SUCCESS = 0
PARAM_INVALID = 103900
TIMEOUT = 103901
NOT_CONNECTED = 103902
ALREADY_CONNECTED = 103903
NOTIFY_FAILED = 103904
UNSUPPORTED = 103905
FAILED = 503900
RESOURCE_EXHAUSTED = 203900

# libstdc++ std::map<ge::AscendString, ge::AscendString> layout: 8 bytes of
# comparator state, then the red-black tree header whose left/right pointers
# reference the header itself in an empty map.
_EMPTY_MAP_SIZE = 48
_EMPTY_MAP_HEADER_OFFSET = 8
_EMPTY_MAP_SELF_POINTER_OFFSET = 24

# libstdc++ std::vector<T> layout: {begin, end, capacity}.
_VECTOR_SIZE = 24

# libstdc++ std::shared_ptr control block (_Sp_counted_base) layout:
# {vtable, use_count@8, weak_count@12}. Virtual slots in Itanium ABI
# declaration order: ~D1@0, ~D0@8, _M_dispose@16, _M_destroy@24.
_SHARED_PTR_CONTROL_OFFSET = 8
_SHARED_PTR_COUNTS_OFFSET = 8
_SHARED_PTR_VTABLE_DISPOSE_SLOT = 16
_SHARED_PTR_VTABLE_DESTROY_SLOT = 24


class MemType(IntEnum):
    MEM_DEVICE = 0
    MEM_HOST = 1


class TransferOp(IntEnum):
    READ = 0
    WRITE = 1


class TransferStatus(IntEnum):
    WAITING = 0
    COMPLETED = 1
    TIMEOUT = 2
    FAILED = 3


class MemDesc(ctypes.Structure):
    """Mirror of CANN 9.1/9.2 ``hixl::MemDesc``, including zeroed reserved bytes."""

    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("len", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * 128),
    ]

    def __init__(self, address: int, size: int) -> None:
        super().__init__()
        self.addr = address
        self.len = size


class TransferOpDesc(ctypes.Structure):
    """Mirror of ``hixl::TransferOpDesc``."""

    _fields_ = [
        ("local_addr", ctypes.c_uint64),
        ("remote_addr", ctypes.c_uint64),
        ("len", ctypes.c_uint64),
    ]


class TransferArgs(ctypes.Structure):
    """Mirror of ``hixl::TransferArgs``; zeroed storage matches the defaults."""

    _fields_ = [
        ("user_data", ctypes.c_void_p),
        ("reserved", ctypes.c_uint8 * 120),
    ]


_HIXL_CTOR = "_ZN4hixl4HixlC1Ev"
_HIXL_DTOR = "_ZN4hixl4HixlD1Ev"
_HIXL_FINALIZE = "_ZN4hixl4Hixl8FinalizeEv"
_HIXL_INITIALIZE = "_ZN4hixl4Hixl10InitializeERKN2ge12AscendStringERKSt3mapIS2_S2_St4lessIS2_ESaISt4pairIS3_S2_EEE"
_HIXL_REGISTER_MEM = "_ZN4hixl4Hixl11RegisterMemERKNS_7MemDescENS_7MemTypeERPv"
_HIXL_DEREGISTER_MEM = "_ZN4hixl4Hixl13DeregisterMemEPv"
_HIXL_CONNECT = "_ZN4hixl4Hixl7ConnectERKN2ge12AscendStringEi"
_HIXL_DISCONNECT = "_ZN4hixl4Hixl10DisconnectERKN2ge12AscendStringEi"
_HIXL_TRANSFER_ASYNC = (
    "_ZN4hixl4Hixl13TransferAsyncERKN2ge12AscendStringENS_10TransferOpERKSt6vectorINS_14"
    "TransferOpDescESaIS7_EERKNS_12TransferArgsERPv"
)
_HIXL_GET_TRANSFER_STATUS = "_ZN4hixl4Hixl17GetTransferStatusERKPvRNS_14TransferStatusE"

_ASCEND_STRING_CTOR = "_ZN2ge12AscendStringC1EPKcm"
_ASCEND_STRING_CTOR_NO_LEN = "_ZN2ge12AscendStringC1EPKc"

_HIXL_LIBRARY_NAMES = ("libcann_hixl.so", "libhixl.so")
_STRING_LIBRARY_NAMES = ("libmetadef.so", "libgraph.so")
_LIBRARY_ARCH_DIRS = ("aarch64-linux", "x86_64-linux")


def _library_search_dirs() -> list[str]:
    dirs: list[str] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        dirs.append(os.path.join(ascend_home, "lib64"))
        dirs.extend(os.path.join(ascend_home, arch, "lib64") for arch in _LIBRARY_ARCH_DIRS)
    dirs.extend(sorted(glob.glob("/usr/local/Ascend/*/lib64")))
    for arch in _LIBRARY_ARCH_DIRS:
        dirs.extend(sorted(glob.glob(f"/usr/local/Ascend/*/{arch}/lib64")))
    return list(dict.fromkeys(dirs))


class _HixlBindings:
    """ctypes views of the exported HIXL engine symbols."""

    def __init__(self) -> None:
        hixl_path, string_path = self._locate_libraries()
        # Load the AscendString provider first so the engine library resolves
        # its imported constructor through the global symbol namespace.
        self._string_lib = ctypes.CDLL(string_path, mode=ctypes.RTLD_GLOBAL)
        self._hixl_lib = ctypes.CDLL(hixl_path, mode=ctypes.RTLD_GLOBAL)
        self.hixl_path = hixl_path

        self.hixl_ctor = self._symbol(self._hixl_lib, _HIXL_CTOR, None, [ctypes.c_void_p])
        self.hixl_dtor = self._symbol(self._hixl_lib, _HIXL_DTOR, None, [ctypes.c_void_p])
        self.hixl_finalize = self._symbol(self._hixl_lib, _HIXL_FINALIZE, None, [ctypes.c_void_p])
        self.initialize = self._symbol(
            self._hixl_lib,
            _HIXL_INITIALIZE,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
        )
        self.register_mem = self._symbol(
            self._hixl_lib,
            _HIXL_REGISTER_MEM,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
        )
        self.deregister_mem = self._symbol(
            self._hixl_lib,
            _HIXL_DEREGISTER_MEM,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.c_void_p],
        )
        self.connect = self._symbol(
            self._hixl_lib,
            _HIXL_CONNECT,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32],
        )
        self.disconnect = self._symbol(
            self._hixl_lib,
            _HIXL_DISCONNECT,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32],
        )
        self.transfer_async = self._symbol(
            self._hixl_lib,
            _HIXL_TRANSFER_ASYNC,
            ctypes.c_uint32,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self.get_transfer_status = self._symbol(
            self._hixl_lib,
            _HIXL_GET_TRANSFER_STATUS,
            ctypes.c_uint32,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32)],
        )
        self.ascend_string_ctor = self._ascend_string_ctor()
        logger.info("HIXL EPLB uses the vllm_ascend ctypes binding for %s.", hixl_path)

    @staticmethod
    def _locate_libraries() -> tuple[str, str]:
        searched: list[str] = []
        for directory in _library_search_dirs():
            for hixl_name in _HIXL_LIBRARY_NAMES:
                hixl_path = os.path.join(directory, hixl_name)
                if not os.path.isfile(hixl_path):
                    searched.append(hixl_path)
                    continue
                for string_name in _STRING_LIBRARY_NAMES:
                    string_path = os.path.join(directory, string_name)
                    if os.path.isfile(string_path):
                        return hixl_path, string_path
                searched.append(f"{hixl_path} without {_STRING_LIBRARY_NAMES} provider")
        raise RuntimeError(
            "No CANN HIXL libraries found. Searched: "
            + (", ".join(searched) if searched else "none")
            + ". Source the CANN environment (set_env.sh) or install a CANN toolkit "
            "with HIXL, or install the official hixl Python package."
        )

    def _ascend_string_ctor(self):
        for symbol in (_ASCEND_STRING_CTOR, _ASCEND_STRING_CTOR_NO_LEN):
            try:
                return self._symbol(
                    self._string_lib,
                    symbol,
                    None,
                    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t],
                )
            except RuntimeError:
                continue
        raise RuntimeError(
            f"Neither {_ASCEND_STRING_CTOR} nor {_ASCEND_STRING_CTOR_NO_LEN} is exported; "
            "the installed CANN toolkit does not match the probed HIXL ABI"
        )

    @staticmethod
    def _symbol(library: ctypes.CDLL, symbol: str, restype, argtypes) -> Callable[..., Any]:
        try:
            function = getattr(library, symbol)
        except AttributeError as error:
            raise RuntimeError(
                f"{symbol} is not exported by the loaded CANN libraries; the installed "
                "HIXL version does not match the ABI this binding was probed against"
            ) from error
        function.restype = restype
        function.argtypes = argtypes
        return function


_BINDINGS: _HixlBindings | None = None
_BINDINGS_LOCK = threading.Lock()


def ensure_available() -> _HixlBindings:
    """Load the toolkit libraries once; raise RuntimeError when unavailable."""

    global _BINDINGS
    with _BINDINGS_LOCK:
        if _BINDINGS is None:
            _BINDINGS = _HixlBindings()
        return _BINDINGS


class _AscendString:
    """A ``ge::AscendString`` built through the libmetadef constructor.

    The shared library owns the reference-counted storage; ``destroy``
    releases it by replicating libstdc++'s ``shared_ptr`` release path. Any
    unexpected control-block state keeps the allocation alive instead of
    risking an invalid free; engine strings are cached per handle, so such a
    leak stays bounded by the peer count.
    """

    __slots__ = ("_storage", "_buffer", "_destroyed")

    def __init__(self, bindings: _HixlBindings, value: str) -> None:
        text = value.encode()
        self._storage: ctypes.Array[ctypes.c_char] | None = ctypes.create_string_buffer(text, len(text) + 1)
        self._buffer = (ctypes.c_char * 16)()
        self._destroyed = False
        bindings.ascend_string_ctor(self._buffer, self._storage, len(text))

    @property
    def buffer(self) -> ctypes.Array:
        return self._buffer

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        control = struct.unpack_from("=Q", self._buffer, _SHARED_PTR_CONTROL_OFFSET)[0]
        if not control:
            return
        control_view = (ctypes.c_char * 16).from_address(control)
        vtable = struct.unpack_from("=Q", control_view, 0)[0]
        use_count, weak_count = struct.unpack_from("=ii", control_view, _SHARED_PTR_COUNTS_OFFSET)
        if not vtable or use_count != 1 or weak_count != 1:
            return
        dispose_address = self._vtable_slot(vtable, _SHARED_PTR_VTABLE_DISPOSE_SLOT)
        destroy_address = self._vtable_slot(vtable, _SHARED_PTR_VTABLE_DESTROY_SLOT)
        if not dispose_address or not destroy_address:
            return
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(dispose_address)(control)
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(destroy_address)(control)
        self._storage = None

    @staticmethod
    def _vtable_slot(vtable: int, offset: int) -> int:
        return struct.unpack_from("=Q", (ctypes.c_char * 8).from_address(vtable + offset))[0]


class Hixl:
    """Engine handle mirroring the official ``hixl.Hixl`` surface used by EPLB."""

    def __init__(self) -> None:
        self._bindings = ensure_available()
        self._lock = threading.Lock()
        self._initialized = False
        self._engine: ctypes.Array[ctypes.c_char] | None = (ctypes.c_char * 8)()
        self._bindings.hixl_ctor(self._engine)
        self._strings: dict[str, _AscendString] = {}

    def _engine_string(self, engine: str) -> _AscendString:
        engine_string = self._strings.get(engine)
        if engine_string is None:
            engine_string = _AscendString(self._bindings, engine)
            self._strings[engine] = engine_string
        return engine_string

    @staticmethod
    def _empty_options_map() -> ctypes.Array:
        options_map = (ctypes.c_char * _EMPTY_MAP_SIZE)()
        header = ctypes.addressof(options_map) + _EMPTY_MAP_HEADER_OFFSET
        struct.pack_into("=QQ", options_map, _EMPTY_MAP_SELF_POINTER_OFFSET, header, header)
        return options_map

    def initialize(self, local_engine: str, options: dict[str, str] | None = None) -> int:
        if options:
            raise ValueError("the ctypes HIXL binding supports no initialize options")
        with self._lock:
            if self._initialized:
                return SUCCESS
            engine_string = self._engine_string(local_engine)
            options_map = self._empty_options_map()
            status = self._bindings.initialize(
                self._engine,
                ctypes.byref(engine_string.buffer),
                ctypes.byref(options_map),
            )
            if status == SUCCESS:
                self._initialized = True
            return status

    def register_mem(self, mem_desc: MemDesc, mem_type: MemType) -> tuple[int, int | None]:
        handle = ctypes.c_void_p()
        with self._lock:
            status = self._bindings.register_mem(
                self._engine,
                ctypes.byref(mem_desc),
                int(mem_type),
                ctypes.byref(handle),
            )
        return status, handle.value

    def deregister_mem(self, mem_handle: int) -> int:
        with self._lock:
            return self._bindings.deregister_mem(self._engine, ctypes.c_void_p(mem_handle))

    def connect(self, remote_engine: str, timeout_in_millis: int = 1000) -> int:
        with self._lock:
            engine_string = self._engine_string(remote_engine)
            return self._bindings.connect(
                self._engine,
                ctypes.byref(engine_string.buffer),
                timeout_in_millis,
            )

    def disconnect(self, remote_engine: str, timeout_in_millis: int = 1000) -> int:
        with self._lock:
            engine_string = self._engine_string(remote_engine)
            return self._bindings.disconnect(
                self._engine,
                ctypes.byref(engine_string.buffer),
                timeout_in_millis,
            )

    def transfer_async(
        self,
        remote_engine: str,
        operation: TransferOp,
        op_descs: list[TransferOpDesc],
    ) -> tuple[int, int | None]:
        count = len(op_descs)
        desc_array = (TransferOpDesc * count)(*op_descs)
        desc_begin = ctypes.addressof(desc_array)
        desc_end = desc_begin + count * ctypes.sizeof(TransferOpDesc)
        desc_vector = (ctypes.c_uint64 * 3)(desc_begin, desc_end, desc_end)
        transfer_args = TransferArgs()
        request = ctypes.c_void_p()
        engine_string = self._engine_string(remote_engine)
        with self._lock:
            status = self._bindings.transfer_async(
                self._engine,
                ctypes.byref(engine_string.buffer),
                int(operation),
                ctypes.byref(desc_vector),
                ctypes.byref(transfer_args),
                ctypes.byref(request),
            )
        return status, request.value

    def get_transfer_status(self, request: int) -> tuple[int, TransferStatus]:
        request_handle = ctypes.c_void_p(request)
        transfer_status = ctypes.c_int32()
        with self._lock:
            status = self._bindings.get_transfer_status(
                self._engine,
                ctypes.byref(request_handle),
                ctypes.byref(transfer_status),
            )
        return status, TransferStatus(transfer_status.value)

    def finalize(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            self._bindings.hixl_finalize(self._engine)
            self._initialized = False

    def _close(self) -> None:
        self.finalize()
        with self._lock:
            if self._engine is None:
                return
            for engine_string in self._strings.values():
                engine_string.destroy()
            self._strings.clear()
            self._bindings.hixl_dtor(self._engine)
            self._engine = None

    def __del__(self) -> None:
        # contextlib may already be cleared during interpreter shutdown.
        try:  # noqa: SIM105
            self._close()
        except Exception:
            pass
