import logging
import threading
import time
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Generic, TypeVar

from .error import RegistrationConflictError, ServiceBusyError, StaleSessionError

logger = logging.getLogger(__name__)

IdentityT = TypeVar("IdentityT", bound=Hashable)
ServiceT = TypeVar("ServiceT")


@dataclass
class _ActiveService(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    service: ServiceT
    last_seen: float


@dataclass(frozen=True)
class _RecoverableSession:
    session_id: str
    fingerprint: bytes


@dataclass(frozen=True)
class _PendingRegistration(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    future: Future[ServiceT]
    restore_on_failure: _RecoverableSession | None = None


@dataclass(frozen=True)
class _ExpiringService(Generic[ServiceT]):
    session_id: str
    fingerprint: bytes
    service: ServiceT


_ServiceState = (
    _PendingRegistration[ServiceT] | _ActiveService[ServiceT] | _ExpiringService[ServiceT] | _RecoverableSession
)


class ServiceLifecycleManager(Generic[IdentityT, ServiceT]):
    """Manage the registration, lease, expiration, and closure of one service type.

    Session-bound requests use ``get_for_session``. Internal lookups use ``find``
    so they do not renew another service's lease. When ``owner_close_handler`` is
    provided, expiration and shutdown delegate service closure through it.

             register()          succeeds          lease expires      close finishes
      +--------+       +-------------+       +--------+       +----------+       +-------------+
      | absent |------>| registering |------>| active |------>| expiring |------>| recoverable |
      +--------+       +-------------+       +--------+       +----------+       +-------------+
          ^                |    ^                                                       |
          | factory fails  |    | register again                                        |
          | without a      |    +-------------------------------------------------------+
          | saved state    |
          +----------------+

    A registration started from ``recoverable`` saves that state until its
    factory succeeds; factory failure returns to ``recoverable`` instead of
    losing the session's recovery path. Registering a new session from ``active``
    retires and closes the old session before moving to ``registering``.

    Unregister retires the session and moves ``active`` or ``recoverable`` to
    ``absent``. During ``expiring``, it makes close completion move to ``absent``
    instead of ``recoverable``. Registration is rejected while a service is
    expiring, and unregister does not interrupt a registration already in progress.

    ``_lock`` serializes every current-state transition and retired-session
    update. An identity therefore has one current state, while retired session
    IDs remain alongside a newer state so stale calls stay rejected.
    ``_expiration_lock`` prevents expiration and shutdown from closing the same
    service concurrently; ``_maintenance_lock`` protects the single background
    maintenance thread.
    """

    def __init__(
        self,
        service_name: str,
        close_service: Callable[[ServiceT], None],
        lease_timeout_s: float,
        check_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        thread_name: str | None = None,
        owner_close_handler: Callable[[IdentityT, ServiceT], None] | None = None,
    ):
        if not service_name:
            raise ValueError("service_name must not be empty")
        if lease_timeout_s <= 0:
            raise ValueError(f"lease_timeout_s must be greater than 0, got {lease_timeout_s}")
        if check_interval_s <= 0:
            raise ValueError(f"check_interval_s must be greater than 0, got {check_interval_s}")

        self._service_name = service_name
        self._close_service = close_service
        self._lease_timeout_s = lease_timeout_s
        self._check_interval_s = check_interval_s
        self._clock = clock
        self._thread_name = thread_name or f"{service_name.lower()}-service-lifecycle"
        self._owner_close_handler = owner_close_handler

        self._lock = threading.RLock()
        self._states: dict[IdentityT, _ServiceState[ServiceT]] = {}
        self._retired_sessions: dict[IdentityT, set[str]] = {}
        self._closed = False

        self._expiration_lock = threading.RLock()
        self._maintenance_lock = threading.Lock()
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    @property
    def count(self) -> int:
        with self._lock:
            return sum(isinstance(state, _ActiveService) for state in self._states.values())

    @property
    def is_running(self) -> bool:
        with self._maintenance_lock:
            return self._maintenance_thread is not None and self._maintenance_thread.is_alive()

    def items(self) -> tuple[tuple[IdentityT, ServiceT], ...]:
        with self._lock:
            return tuple(
                (identity, state.service)
                for identity, state in self._states.items()
                if isinstance(state, _ActiveService)
            )

    # ==============================
    # One registration per identity
    # ==============================

    # Registration is one state transition per identity. Repeated calls for the
    # same session share its result; a different session first makes the old one
    # invalid and removes its service before constructing the replacement.
    # Factories and close calls run outside the lifecycle lock, and a result is
    # published only while its registration is current and the manager is open.

    def register(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        factory: Callable[[], ServiceT],
    ) -> ServiceT:
        self._validate_session_id(session_id)
        old_service = None
        registration = None

        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            state = self._states.get(identity)
            if isinstance(state, _ExpiringService):
                raise ServiceBusyError(f"{self._service_name} {identity!r} is being expired")

            if isinstance(state, _ActiveService):
                if state.session_id == session_id:
                    self._validate_fingerprint(identity, state.fingerprint, fingerprint)
                    state.last_seen = self._clock()
                    return state.service

                self._retire_session_locked(identity, state.session_id)
                old_service = state.service
                state = None

            if isinstance(state, _PendingRegistration):
                if state.session_id != session_id:
                    raise RegistrationConflictError(
                        f"{self._service_name} {identity!r} is already registering session {state.session_id!r}"
                    )
                self._validate_fingerprint(identity, state.fingerprint, fingerprint)
                wait_future = state.future
            else:
                restore_on_failure = None
                if isinstance(state, _RecoverableSession):
                    if state.session_id == session_id:
                        self._validate_fingerprint(identity, state.fingerprint, fingerprint)
                        restore_on_failure = state
                    else:
                        self._retire_session_locked(identity, state.session_id)
                elif state is not None:
                    raise TypeError(f"Unexpected {self._service_name} lifecycle state: {type(state).__name__}")

                registration = _PendingRegistration(session_id, fingerprint, Future(), restore_on_failure)
                self._states[identity] = registration
                wait_future = None

        if wait_future is not None:
            return wait_future.result()
        assert registration is not None
        return self._create_and_publish(identity, session_id, fingerprint, factory, registration, old_service)

    def _create_and_publish(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        factory: Callable[[], ServiceT],
        registration: _PendingRegistration[ServiceT],
        old_service: ServiceT | None,
    ) -> ServiceT:
        service = None
        try:
            if old_service is not None:
                self._close_service(old_service)

            service = factory()
            with self._lock:
                self._publish_locked(identity, session_id, fingerprint, registration, service)
        except BaseException as exc:
            self._fail_registration(identity, registration, service, exc)
            raise

        registration.future.set_result(service)
        return service

    def _publish_locked(
        self,
        identity: IdentityT,
        session_id: str,
        fingerprint: bytes,
        registration: _PendingRegistration[ServiceT],
        service: ServiceT,
    ) -> None:
        self._raise_if_closed()
        if self._states.get(identity) is not registration:
            raise RuntimeError(f"{self._service_name} {identity!r} registration is no longer active")

        # Reinsert the identity so active iteration retains publication order,
        # including when concurrent factories finish out of order.
        del self._states[identity]
        self._states[identity] = _ActiveService(session_id, fingerprint, service, self._clock())

    def _fail_registration(
        self,
        identity: IdentityT,
        registration: _PendingRegistration[ServiceT],
        service: ServiceT | None,
        exc: BaseException,
    ) -> None:
        should_complete_registration = False
        with self._lock:
            if self._states.get(identity) is registration:
                if registration.restore_on_failure is None:
                    del self._states[identity]
                else:
                    self._states[identity] = registration.restore_on_failure
                should_complete_registration = True

        if service is not None:
            self._close_service_safely(service)
        if should_complete_registration:
            registration.future.set_exception(exc)

    # ==============================
    # Session access and release
    # ==============================

    # The lifecycle lock decides which session owns each service and makes that
    # decision visible before slow backend cleanup begins. Only requests for the
    # current session renew its lease; server-internal reads do not. These rules
    # keep session replacement, expiration, and unregister consistent while
    # allowing unrelated lifecycle work to continue during cleanup.

    def renew(self, identity: IdentityT, session_id: str) -> bool:
        return self._get_and_renew_entry(identity, session_id) is not None

    def find(self, identity: IdentityT) -> ServiceT | None:
        """Return a service without validating or renewing its session."""
        with self._lock:
            self._raise_if_closed()
            state = self._states.get(identity)
            return state.service if isinstance(state, _ActiveService) else None

    def get_for_session(self, identity: IdentityT, session_id: str) -> ServiceT | None:
        """Validate the session, renew its lease, and return the service."""
        entry = self._get_and_renew_entry(identity, session_id)
        return None if entry is None else entry.service

    def _get_and_renew_entry(self, identity: IdentityT, session_id: str) -> _ActiveService[ServiceT] | None:
        """Validate, resolve, and renew a session atomically.

        These steps stay in one method and share the lifecycle lock so expiration
        cannot detach the service between session validation and lease renewal.
        """
        self._validate_session_id(session_id)
        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            state = self._states.get(identity)
            if not isinstance(state, _ActiveService):
                return None
            self._validate_session(identity, session_id, state.session_id)
            state.last_seen = self._clock()
            return state

    def unregister(self, identity: IdentityT, session_id: str) -> bool:
        """Retire a session and release its service when this call owns cleanup.

        An active service is detached under the lifecycle lock and closed after
        that lock is released. Expiration already owns the close of an expiring
        service, so unregister only retires its session and leaves the state in
        place until cleanup finishes; this prevents replacement from racing the
        old close. A recoverable session has no service left and only loses its
        recovery permission. A pending factory may already be running outside
        the lock and cannot be cancelled safely, while an absent identity has
        nothing to unregister; both return ``False``.
        """
        self._validate_session_id(session_id)
        service = None

        with self._lock:
            self._raise_if_closed()
            self._raise_if_retired(identity, session_id)
            state = self._states.get(identity)

            if isinstance(state, _PendingRegistration):
                self._validate_session(identity, session_id, state.session_id)
                return False

            if state is None:
                return False

            if isinstance(state, _ExpiringService):
                self._validate_session(identity, session_id, state.session_id)
                self._retire_session_locked(identity, session_id)
                return True

            if isinstance(state, _RecoverableSession):
                self._validate_session(identity, session_id, state.session_id)
                del self._states[identity]
                self._retire_session_locked(identity, session_id)
                return True

            if not isinstance(state, _ActiveService):
                raise TypeError(f"Unexpected {self._service_name} lifecycle state: {type(state).__name__}")
            self._validate_session(identity, session_id, state.session_id)

            del self._states[identity]
            self._retire_session_locked(identity, session_id)
            service = state.service

        self._close_service(service)
        return True

    def _retire_session_locked(self, identity: IdentityT, session_id: str) -> None:
        self._retired_sessions.setdefault(identity, set()).add(session_id)

    # ==============================
    # Lease expiration and maintenance
    # ==============================

    # Expiration first makes stale services unavailable by moving them to
    # ``expiring`` under the lifecycle lock. It then closes them through the
    # owner close handler without that lock and exposes the session as
    # recoverable only after cleanup finishes. The maintenance thread runs this
    # same transaction.

    def expire_leases(self) -> int:
        with self._expiration_lock:
            stale_before = self._clock() - self._lease_timeout_s
            with self._lock:
                if self._closed:
                    return 0
                expiring_services = [
                    (
                        identity,
                        _ExpiringService(state.session_id, state.fingerprint, state.service),
                    )
                    for identity, state in self._states.items()
                    if isinstance(state, _ActiveService) and state.last_seen <= stale_before
                ]
                for identity, expiring in expiring_services:
                    self._states[identity] = expiring

            for identity, expiring in expiring_services:
                self._close_on_owner_safely(identity, expiring.service)
                self._finish_expiration(identity, expiring)
            return len(expiring_services)

    def _finish_expiration(self, identity: IdentityT, expiring: _ExpiringService[ServiceT]) -> None:
        with self._lock:
            if self._states.get(identity) is not expiring:
                return
            if expiring.session_id in self._retired_sessions.get(identity, ()):
                del self._states[identity]
            else:
                self._states[identity] = _RecoverableSession(expiring.session_id, expiring.fingerprint)

    def start_maintenance(self) -> None:
        with self._lock:
            self._raise_if_closed()
            with self._maintenance_lock:
                if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
                    return

                self._maintenance_stop.clear()
                self._maintenance_thread = threading.Thread(
                    target=self._maintenance_loop, daemon=True, name=self._thread_name
                )
                self._maintenance_thread.start()

    def stop_maintenance(self, wait: bool = True) -> None:
        with self._maintenance_lock:
            thread = self._maintenance_thread
            if thread is None:
                return
            self._maintenance_stop.set()

        if not wait:
            return
        if thread is not threading.current_thread():
            thread.join()

        with self._maintenance_lock:
            if self._maintenance_thread is thread:
                self._maintenance_thread = None

    def _maintenance_loop(self) -> None:
        while not self._maintenance_stop.wait(self._check_interval_s):
            try:
                self.expire_leases()
            except Exception:
                logger.exception("%s service lifecycle maintenance failed", self._service_name)

    # ==============================
    # Shutdown and service closure
    # ==============================

    # Closing rejects new lifecycle work before joining the maintenance thread.
    # Entries are removed under lock and then closed without holding it. Each
    # service gets one close attempt because repeating backend close may be unsafe.

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self.stop_maintenance()
        with self._expiration_lock, self._lock:
            states = tuple(self._states.items())
            services = [
                (identity, state.service)
                for identity, state in states
                if isinstance(state, (_ActiveService, _ExpiringService))
            ]
            registrations = [state for _, state in states if isinstance(state, _PendingRegistration)]
            self._states.clear()
            self._retired_sessions.clear()

        # Wake every caller waiting for registration with a close error. A factory
        # already running outside the lock cannot be cancelled and will clean up
        # its result when it eventually returns.
        for registration in registrations:
            registration.future.set_exception(RuntimeError(f"{self._service_name} lifecycle manager is closed"))

        for identity, service in services:
            self._close_on_owner_safely(identity, service)

    def _close_service_safely(self, service: ServiceT) -> None:
        try:
            self._close_service(service)
        except Exception:
            logger.exception("Failed to close %s service %r", self._service_name, service)

    def _close_on_owner_safely(self, identity: IdentityT, service: ServiceT) -> None:
        if self._owner_close_handler is None:
            self._close_service_safely(service)
            return

        try:
            self._owner_close_handler(identity, service)
        except Exception:
            logger.exception("Failed to close %s service %r on its owner", self._service_name, service)

    # ==============================
    # Session and registration validation
    # ==============================

    # Invalid local arguments raise ordinary value errors. Stale sessions and
    # conflicting registration data use lifecycle errors so callers can handle
    # them separately.

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self._service_name} lifecycle manager is closed")

    def _raise_if_retired(self, identity: IdentityT, session_id: str) -> None:
        if session_id in self._retired_sessions.get(identity, ()):
            raise StaleSessionError(f"{self._service_name} {identity!r} session {session_id!r} has been retired")

    def _validate_session(self, identity: IdentityT, incoming: str, current: str) -> None:
        if incoming != current:
            raise StaleSessionError(
                f"{self._service_name} {identity!r} session {incoming!r} is stale; current session is {current!r}"
            )

    def _validate_fingerprint(self, identity: IdentityT, existing: bytes, incoming: bytes) -> None:
        if existing != incoming:
            raise RegistrationConflictError(
                f"{self._service_name} {identity!r} is already registered with different configuration"
            )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str):
            raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
        if not session_id:
            raise ValueError("session_id must not be empty")
