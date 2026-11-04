from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import regex as re
import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (
    LOOKUP_MSG,
    RESET_MSG,
    RESP_ERR,
    RESP_OK,
)
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

# One wall-clock budget per lookup() call, covering send, reply wait and reply
# validation. 0 or negative never means "wait forever": it is rejected at
# construction so a typo cannot silently disable the bound.
DEFAULT_TIMEOUT_MS = 5000
WARN_EVERY_N = 100
SHUTDOWN_JOIN_TIMEOUT_S = 2.0

ENVELOPE_VERSION = 1
INT_FRAME_BYTES = 4
EXPECTED_LOOKUP_FRAMES = 5
MAX_HASH_CHARS = 1024
MAX_HASHES = 1_048_576
MAX_GROUPS = 4096

LOOKUP_STATUS_OK = "ok"
LOOKUP_STATUS_ERROR = "error"

# Poll granularity, kept well below the shortest supported budget so a close()
# is noticed promptly without spinning.
SERVER_POLL_INTERVAL_MS = 100

_OK_FIELDS = frozenset({"version", "status", "hit_tokens"})
_ERROR_FIELDS = frozenset({"version", "status", "code", "message"})


class ProtocolViolation(ValueError):
    """A frame or envelope that must never be read as a hit count."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_MAX_SOCKOPT_MS = 2**31 - 1


def validate_timeout_ms(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"lookup_rpc_timeout_ms must be a positive int of milliseconds, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(
            f"lookup_rpc_timeout_ms must be a positive int of milliseconds, got {value}; "
            "0 and negative values are not treated as an unbounded wait"
        )
    if value > _MAX_SOCKOPT_MS:
        # Python ints are unbounded, but a deadline this large cannot be carried
        # by the socket option and turns the bound back into an unbounded wait --
        # the thing this parameter exists to prevent.
        raise ValueError(
            f"lookup_rpc_timeout_ms must not exceed {_MAX_SOCKOPT_MS} ms, got {value}; "
            "a value that cannot be honoured as a deadline is not accepted"
        )
    return value


def classify_zmq_error(exc: zmq.ZMQError, stage: str) -> str:  # type: ignore
    """Map a socket error onto one closed-set code, keyed by root cause.

    EFSM means the REQ/REP state machine was violated no matter which call site
    surfaced it, and EAGAIN from send means the budget was already spent. Both
    would be invisible if they landed in the generic transport bucket. The errno
    is read from ``zmq.constants`` because ZMQ keeps its own error range
    (156384714+) that does not line up with the OS ``errno`` module.
    """
    if isinstance(exc, zmq.Again):  # type: ignore[attr-defined]
        return f"{stage}:timeout"
    if exc.errno == int(zmq.constants.EFSM):  # type: ignore[attr-defined]
        return f"{stage}:req_state"
    return f"{stage}:transport"


# ZMQ takes the socket-option value as a 32-bit int; validate_timeout_ms already
# caps the configured value, so the only remaining job here is to keep a
# sub-millisecond remainder from becoming a zero-timeout option.
def _sockopt_ms(value: int) -> int:
    return max(1, value)


def _ok_envelope(hit_tokens: int) -> dict[str, Any]:
    return {"version": ENVELOPE_VERSION, "status": LOOKUP_STATUS_OK, "hit_tokens": hit_tokens}


def _error_envelope(code: str, message: str) -> dict[str, Any]:
    return {
        "version": ENVELOPE_VERSION,
        "status": LOOKUP_STATUS_ERROR,
        "code": code,
        "message": _sanitize(message),
    }


_HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")


def _sanitize(message: object) -> str:
    """Keep replies and logs free of key material.

    A backend exception may embed the very key it failed on, so runs that look
    like block hashes are redacted before the message leaves the process. Binary
    payloads are never echoed at all: ``str(b"...")`` yields the *escaped* repr,
    which still carries every key byte, so the hex-run rule alone would leak a
    hash that arrives as bytes (a wire frame, not a log message).
    """
    if isinstance(message, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(message))}-byte binary frame>"
    collapsed = " ".join(str(message).split())
    collapsed = _HEX_RUN.sub("<redacted>", collapsed)
    return collapsed[:160]


def _request_counts(token_len: int, hashes: list[Any], groups: list[Any]) -> str:
    return f"token_len={token_len} hash_count={len(hashes)} group_count={len(groups)}"


def decode_hit_tokens(payload: Any, token_len: int) -> int:
    """Validate one response envelope; raise ProtocolViolation on anything else."""
    if not isinstance(payload, dict):
        raise ProtocolViolation("decode:decode_error")
    fields = frozenset(payload)
    if fields not in (_OK_FIELDS, _ERROR_FIELDS):
        raise ProtocolViolation("decode:field_invalid")
    version = payload.get("version")
    if type(version) is not int or version != ENVELOPE_VERSION:
        raise ProtocolViolation("decode:decode_error")
    status = payload.get("status")
    if status == LOOKUP_STATUS_ERROR:
        code = payload.get("code")
        if not isinstance(code, str):
            raise ProtocolViolation("decode:field_invalid")
        raise ProtocolViolation(f"server:{code}")
    if status != LOOKUP_STATUS_OK:
        raise ProtocolViolation("decode:field_invalid")
    hit_tokens = payload.get("hit_tokens")
    if type(hit_tokens) is not int or hit_tokens < 0 or hit_tokens > token_len:
        raise ProtocolViolation("decode:field_invalid")
    return hit_tokens


def get_zmq_rpc_path_lookup(vllm_config: VllmConfig) -> str:
    # NOTE: This isolates lookup IPC paths only. MooncakeConnector (V1) worker
    # handshake ports and MooncakeConnectorV2 scheduler control ports may still
    # overlap ports reserved by other workers or components. DP indices do not
    # guarantee disjoint TCP port ranges.
    # Dense engines reset their compute DP rank, but retain their serving index.
    dp_index = vllm_config.parallel_config.data_parallel_index
    base_url = envs.VLLM_RPC_BASE_PATH
    rpc_port = 0
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    if "lookup_rpc_port" in extra_config:
        rpc_port = extra_config["lookup_rpc_port"]
    elif "mooncake_rpc_port" in extra_config:
        rpc_port = extra_config["mooncake_rpc_port"]
        logger.warning(
            "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
        )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_index}"


class LookupKeyClient:
    """Scheduler-side REQ client that fails open to "no external hit tokens".

    A transport fault must degrade to a recomputed prefix -- never to a crash and
    never to a permanently dead socket. The old behaviour closed the client on
    the first timeout, so every later lookup answered 0 without saying so;
    recovery therefore happens inside the failing call.
    """

    def __init__(self, vllm_config: VllmConfig, clock: Any = time.monotonic):
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder()
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.timeout_ms = validate_timeout_ms(extra_config.get("lookup_rpc_timeout_ms", DEFAULT_TIMEOUT_MS))
        self._clock = clock
        self._budget_start = clock()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = self._open_socket()
        self._closed = False
        self.failure_counts: dict[str, int] = {}

    def _open_socket(self) -> Any:
        socket = make_zmq_socket(
            self.ctx,
            self.socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        socket.setsockopt(zmq.SNDTIMEO, _sockopt_ms(self.timeout_ms))  # type: ignore[attr-defined]
        socket.setsockopt(zmq.RCVTIMEO, _sockopt_ms(self.timeout_ms))  # type: ignore[attr-defined]
        return socket

    @property
    def closed(self) -> bool:
        return self._closed

    def snapshot(self) -> dict[str, int]:
        return dict(self.failure_counts)

    def _start_budget(self) -> None:
        self._budget_start = self._clock()

    def _elapsed_us(self) -> int:
        return max(0, int((self._clock() - self._budget_start) * 1_000_000))

    def _remaining_ms(self) -> int:
        # Whole-microsecond arithmetic: flooring a float millisecond difference
        # can hand back one millisecond more than the caller ever budgeted for.
        return (self.timeout_ms * 1000 - self._elapsed_us()) // 1000

    def _elapsed_ms(self) -> int:
        return self._elapsed_us() // 1000

    def _note_failure(self, code: str, detail: str) -> None:
        count = self.failure_counts.get(code, 0) + 1
        self.failure_counts[code] = count
        if count == 1 or count % WARN_EVERY_N == 0:
            logger.warning(
                "KV pool lookup RPC failed; reporting 0 external hit tokens code=%s stage=%s elapsed_ms=%d %s total=%d",
                code,
                code.split(":", 1)[0],
                self._elapsed_ms(),
                detail,
                count,
            )

    def _recover(self) -> None:
        retiring = self.socket
        self.socket = None  # type: ignore[assignment]
        try:
            retiring.close(linger=0)
        except Exception:  # pragma: no cover - retirement must not mask the fail-open
            logger.debug("KV pool lookup REQ retirement raised", exc_info=True)
        try:
            self.socket = self._open_socket()
        except Exception as exc:
            # Recovery sits inside the fail-open path, so a failed reopen (EMFILE on a
            # process out of descriptors) must not become an exception in the scheduler.
            # The handle stays absent and the next call retries, counted either way.
            self._note_failure("recover:open_error", f"op=recover {type(exc).__name__}")
            logger.warning("KV pool lookup REQ reopen failed; lookups stay fail-open until it succeeds")

    def _fail(self, code: str, detail: str) -> int:
        self._note_failure(code, detail)
        self._recover()
        return 0

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        if self._closed:
            return 0

        self._start_budget()
        groups = kv_cache_group_ids or [0]
        detail = _request_counts(token_len, block_hashes, groups)
        if self.socket is None:
            # Only reachable when a previous reopen failed; `_fail` retries the open here.
            return self._fail("send:socket_unavailable", detail)
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(groups)
        all_frames = [
            LOOKUP_MSG,
            token_len.to_bytes(INT_FRAME_BYTES, byteorder="big"),
            *kv_group_frames,
            hbm_hit_tokens.to_bytes(INT_FRAME_BYTES, byteorder="big"),
            *hash_frames,
        ]

        send_budget_ms = self._remaining_ms()
        if send_budget_ms <= 0:
            # The hash conversion and encoding alone spent the budget. Sending now would
            # arm a second full wait behind work that already blew the deadline, so the
            # bound is held by not sending at all.
            return self._fail("send:timeout", detail)
        # Re-armed per call: SNDTIMEO set once at socket creation would let a blocked
        # send wait the full timeout *after* the encode already spent part of it.
        self.socket.setsockopt(zmq.SNDTIMEO, _sockopt_ms(send_budget_ms))  # type: ignore[attr-defined]

        try:
            self.socket.send_multipart(all_frames, copy=False)
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            return self._fail(classify_zmq_error(exc, "send"), detail)

        remaining_ms = self._remaining_ms()
        if remaining_ms <= 0:
            return self._fail("recv:timeout", detail)
        self.socket.setsockopt(zmq.RCVTIMEO, _sockopt_ms(remaining_ms))  # type: ignore[attr-defined]
        try:
            resp_frames = self.socket.recv_multipart(copy=False)
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            return self._fail(classify_zmq_error(exc, "recv"), detail)

        try:
            payload = self._decode_response(resp_frames)
            hit_tokens = decode_hit_tokens(payload, token_len)
        except ProtocolViolation as viol:
            return self._fail(viol.code, detail)
        except Exception:  # pragma: no cover - decoder faults stay fail-open
            logger.debug("KV pool lookup response decode raised", exc_info=True)
            return self._fail("decode:decode_error", detail)

        # Validation is inside the same budget; a reply that only becomes usable
        # after the deadline still counts as a timeout.
        if self._remaining_ms() <= 0:
            return self._fail("recv:timeout", detail)
        return hit_tokens

    def _decode_response(self, resp_frames: Any) -> Any:
        frames = [bytes(frame) for frame in resp_frames]
        if len(frames) != 1:
            raise ProtocolViolation("decode:frame_count")
        if not frames[0]:
            raise ProtocolViolation("decode:decode_error")
        return self.decoder.decode(frames)

    def reset(self) -> bool:
        """Ask rank 0 to drain queued puts, then ``remove_all(force=True)``.

        Shares the lookup budget and the recovery rule; the reply keeps the bare
        RESP_OK/RESP_ERR wire format so no other caller has to change.
        """
        if self._closed:
            return False

        self._start_budget()
        if self.socket is None:
            self._note_failure("reset:socket_unavailable", "op=reset")
            self._recover()
            return False
        try:
            self.socket.send(RESET_MSG)
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            self._note_failure(classify_zmq_error(exc, "reset"), "op=reset")
            self._recover()
            return False

        remaining_ms = self._remaining_ms()
        if remaining_ms <= 0:
            self._note_failure("reset:timeout", "op=reset")
            self._recover()
            return False
        self.socket.setsockopt(zmq.RCVTIMEO, _sockopt_ms(remaining_ms))  # type: ignore[attr-defined]
        try:
            resp = self.socket.recv()
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            self._note_failure(classify_zmq_error(exc, "reset"), "op=reset")
            self._recover()
            return False

        if self._remaining_ms() <= 0:
            self._note_failure("reset:timeout", "op=reset")
            self._recover()
            return False
        return bytes(resp) == RESP_OK

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.socket is not None:
                self.socket.close(linger=0)
        finally:
            # Each client owns a private context. Close it too so repeated
            # lookup failures do not leak ZeroMQ IO threads.
            self.ctx.term()


class LookupKeyServer:
    """Worker-side REP server: one reply per received request.

    A malformed request is answered with a versioned error envelope and the loop
    keeps serving. Only an unrecoverable REP socket forces a rebuild, and that
    rebuild happens inside the loop instead of ending the thread.
    """

    def __init__(
        self,
        pool_worker: KVPoolWorker,
        vllm_config: VllmConfig,
    ):
        self.decoder = MsgpackDecoder()
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        self.socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = self._open_socket()

        self.pool_worker = pool_worker
        self.running = True
        self._closed = False
        self.failure_counts: dict[str, int] = {}
        # Woken by close() so a pass sitting in the reopen backoff returns at once
        # instead of paying the interval once more on the way out.
        self._reopen_backoff = threading.Event()

        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _open_socket(self) -> Any:
        return make_zmq_socket(
            self.ctx,
            self.socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

    def snapshot(self) -> dict[str, int]:
        return dict(self.failure_counts)

    def _note_failure(self, code: str, detail: str) -> None:
        count = self.failure_counts.get(code, 0) + 1
        self.failure_counts[code] = count
        if count == 1 or count % WARN_EVERY_N == 0:
            logger.warning(
                "KV pool lookup RPC request failed code=%s stage=%s %s total=%d",
                code,
                code.split(":", 1)[0],
                detail,
                count,
            )

    def _rebuild_socket(self, poller: zmq.Poller) -> None:  # type: ignore
        retiring = self.socket
        self.socket = None  # type: ignore[assignment]
        # Unregister before closing: a handle that is already gone makes
        # unregister raise ENOTSOCK, and letting that escape would leave the
        # loop polling a dead endpoint forever.
        try:
            poller.unregister(retiring)
        except Exception:  # pragma: no cover - best effort on a broken handle
            logger.debug("KV pool lookup REP unregister raised", exc_info=True)
        try:
            retiring.close(linger=0)
        except Exception:  # pragma: no cover - rebuild stays bounded
            logger.debug("KV pool lookup REP retirement raised", exc_info=True)
        try:
            self.socket = self._open_socket()
        except Exception as exc:
            # Nothing can be polled without a handle, and swallowing this quietly would
            # leave the loop polling an empty set -- thread alive, server deaf. Count it
            # and let the next pass retry the open.
            self.socket = None  # type: ignore[assignment]
            # `_note_failure` already reports the first occurrence and every
            # WARN_EVERY_N after it; a second unthrottled line beside it turns a
            # persistent reopen failure into one log line per pass.
            self._note_failure("server:open_error", f"op=rebuild {type(exc).__name__}")
            return
        poller.register(self.socket, zmq.POLLIN)  # type: ignore[attr-defined]

    def _encode(self, envelope: dict[str, Any]) -> list[bytes]:
        frames = [bytes(frame) for frame in self.encoder.encode(envelope)]
        if len(frames) != 1:
            raise ProtocolViolation("decode:frame_count")
        return frames

    @staticmethod
    def _read_int_frame(frame: Any) -> int:
        raw = bytes(frame)
        if len(raw) != INT_FRAME_BYTES:
            raise ProtocolViolation("decode:field_invalid")
        return int.from_bytes(raw, byteorder="big")

    def _decode_list(self, frame: Any) -> Any:
        try:
            return self.decoder.decode([bytes(frame)])
        except ProtocolViolation:
            raise
        except Exception as exc:
            raise ProtocolViolation("decode:decode_error") from exc

    def _handle_lookup(self, all_frames: list[Any]) -> list[bytes]:
        if len(all_frames) != EXPECTED_LOOKUP_FRAMES:
            raise ProtocolViolation("decode:frame_count")
        token_len = self._read_int_frame(all_frames[1])
        kv_group_ids = self._decode_list(all_frames[2])
        hbm_hit_tokens = self._read_int_frame(all_frames[3])
        hashes_str = self._decode_list(all_frames[4])

        if not isinstance(kv_group_ids, list) or not 1 <= len(kv_group_ids) <= MAX_GROUPS:
            raise ProtocolViolation("decode:field_invalid")
        if any(type(gid) is not int or gid < 0 for gid in kv_group_ids):
            raise ProtocolViolation("decode:field_invalid")
        if len(set(kv_group_ids)) != len(kv_group_ids):
            raise ProtocolViolation("decode:field_invalid")
        if not isinstance(hashes_str, list) or not 1 <= len(hashes_str) <= MAX_HASHES:
            raise ProtocolViolation("decode:field_invalid")
        if any(not isinstance(h, str) or not h or len(h) > MAX_HASH_CHARS for h in hashes_str):
            raise ProtocolViolation("decode:field_invalid")
        if hbm_hit_tokens > token_len:
            raise ProtocolViolation("decode:field_invalid")

        detail = _request_counts(token_len, hashes_str, kv_group_ids)
        try:
            result = self.pool_worker.lookup_scheduler(
                token_len,
                hashes_str,
                kv_group_ids,
                use_layerwise=False,
                hbm_hit_tokens=hbm_hit_tokens,
            )
        except Exception as exc:
            self._note_failure("server:backend_error", detail)
            logger.debug("KV pool lookup backend raised", exc_info=True)
            return self._encode(_error_envelope("backend_error", _sanitize(exc)))

        # A zero from the backend is a real miss; only an out-of-range or
        # non-integer answer is a protocol fault worth reporting as one.
        if type(result) is not int or result < 0 or result > token_len:
            self._note_failure("server:field_invalid", detail)
            return self._encode(_error_envelope("field_invalid", "backend hit count out of range"))

        logger.debug(
            "KV pool lookup response token_len=%d group_count=%d hit_tokens=%d",
            token_len,
            len(kv_group_ids),
            result,
        )
        return self._encode(_ok_envelope(result))

    def _handle_reset(self, all_frames: list[Any]) -> list[bytes]:
        if len(all_frames) != 1:
            raise ProtocolViolation("decode:frame_count")
        try:
            self.pool_worker.reset_store()
        except Exception as exc:
            self._note_failure("reset:backend_error", "op=reset")
            logger.warning("AscendStore reset via remove_all failed: %s", _sanitize(exc))
            return [RESP_ERR]
        logger.info("AscendStore reset via remove_all succeeded.")
        return [RESP_OK]

    def _serve_iteration(self, poller: zmq.Poller) -> None:  # type: ignore
        if self.socket is None:
            # Set only by a rebuild whose open failed; polling now would block on an
            # empty registration set, so this pass spends itself retrying the open.
            # Returning straight away made the loop re-enter at full speed while the
            # open kept failing (descriptor exhaustion, the endpoint still bound by a
            # dying worker), burning a core and one log line per pass.
            self._rebuild_socket(poller)
            if self.socket is None:
                self._reopen_backoff.wait(SERVER_POLL_INTERVAL_MS / 1000)
            return
        try:
            readable = poller.poll(timeout=SERVER_POLL_INTERVAL_MS)
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            if not self.running:
                return
            # A poll fault means the registered handle is gone (e.g. the REP was
            # closed underneath the loop). Without a rebuild here the loop keeps
            # polling a dead endpoint and the server goes permanently deaf while
            # its thread stays alive -- the contract's "unrecoverable REP" case.
            self._note_failure(classify_zmq_error(exc, "server"), "op=poll")
            logger.debug("KV pool lookup poll failed; rebuilding REP handle", exc_info=True)
            self._rebuild_socket(poller)
            return
        if not readable:
            return

        try:
            all_frames = list(self.socket.recv_multipart(copy=False))
        except zmq.ZMQError as exc:  # type: ignore[attr-defined]
            if not self.running:
                return
            if isinstance(exc, zmq.Again):  # type: ignore[attr-defined]
                # A poll/spurious timeout is not a protocol-state break: keep the
                # same REP handle.
                return
            self._note_failure("server:transport", "op=recv")
            logger.debug("Failed to receive KV pool lookup request", exc_info=True)
            self._rebuild_socket(poller)
            return

        if not all_frames:
            self._note_failure("decode:frame_count", "frames=0")
            response = self._encode(_error_envelope("frame_count", "empty request"))
        else:
            msg_type = bytes(all_frames[0])
            detail = f"frames={len(all_frames)}"
            try:
                if msg_type == LOOKUP_MSG:
                    response = self._handle_lookup(all_frames)
                elif msg_type == RESET_MSG:
                    response = self._handle_reset(all_frames)
                else:
                    logger.warning(
                        "LookupKeyServer received unknown msg_type len=%d value=%s",
                        len(msg_type),
                        _sanitize(msg_type),
                    )
                    self._note_failure("decode:field_invalid", detail)
                    response = self._encode(_error_envelope("field_invalid", "unknown msg_type"))
            except ProtocolViolation as viol:
                self._note_failure(viol.code, detail)
                response = self._encode(
                    _error_envelope(viol.code.split(":", 1)[1], "request rejected by protocol validation")
                )
            except Exception as exc:  # pragma: no cover - loop must survive the unexpected
                self._note_failure("unexpected:internal_error", detail)
                logger.error("Error handling lookup request: %s", _sanitize(exc))
                response = self._encode(_error_envelope("internal_error", "internal lookup failure"))

        try:
            self.socket.send_multipart(response, copy=False)
        except zmq.ZMQError:  # type: ignore[attr-defined]
            if self.running:
                self._note_failure("server:transport", "op=send")
                logger.debug("Failed to send KV pool lookup response", exc_info=True)
                self._rebuild_socket(poller)

    def _serve(self) -> None:
        poller = zmq.Poller()  # type: ignore[attr-defined]
        poller.register(self.socket, zmq.POLLIN)  # type: ignore[attr-defined]
        try:
            while self.running:
                try:
                    self._serve_iteration(poller)
                except Exception as exc:  # pragma: no cover - one bad request must not end the loop
                    self._note_failure("unexpected:internal_error", "stage=loop")
                    logger.error("KV pool lookup server iteration raised: %s", _sanitize(exc))
        finally:
            if self.socket is not None:
                try:
                    poller.unregister(self.socket)
                except Exception:  # pragma: no cover - shutdown is best effort
                    logger.debug("KV pool lookup server unregister raised", exc_info=True)
                self.socket.close(linger=0)
            self.ctx.term()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.running = False
        self._reopen_backoff.set()
        self.thread.join(SHUTDOWN_JOIN_TIMEOUT_S)
        if self.thread.is_alive():
            logger.warning(
                "KV pool lookup server thread still running after %.1f s; an in-flight "
                "backend call keeps it alive, and the daemon flag keeps the process exiting",
                SHUTDOWN_JOIN_TIMEOUT_S,
            )
