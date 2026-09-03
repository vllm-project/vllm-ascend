import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import (
    RegistrationConflictError,
    ServiceBusyError,
    ServiceLifecycleManager,
    StaleSessionError,
)

LIFECYCLE_MODULE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service.lifecycle"


class _FakeService:
    def __init__(self, closed: threading.Event | None = None):
        self._closed = closed
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        if self._closed is not None:
            self._closed.set()


class _BlockingCloseService(_FakeService):
    def __init__(self, close_started: threading.Event, release_close: threading.Event):
        super().__init__()
        self._close_started = close_started
        self._release_close = release_close

    def close(self) -> None:
        super().close()
        self._close_started.set()
        if not self._release_close.wait(5):
            raise TimeoutError("Timed out waiting to release service close")


class _FailingCloseService(_FakeService):
    def close(self) -> None:
        super().close()
        raise RuntimeError("service close failed")


def _create_manager(
    clock=lambda: 0.0,
    lease_timeout_s: float = 10.0,
    check_interval_s: float = 0.01,
    owner_close_handler: Callable[[str, _FakeService], None] | None = None,
) -> ServiceLifecycleManager[str, _FakeService]:
    return ServiceLifecycleManager(
        "Test",
        lambda service: service.close(),
        lease_timeout_s=lease_timeout_s,
        check_interval_s=check_interval_s,
        clock=clock,
        owner_close_handler=owner_close_handler,
    )


def test_find_is_read_only_and_get_for_session_renews_lease() -> None:
    now = [10.0]
    manager = _create_manager(lambda: now[0])
    service = manager.register("service-0", "session-0", b"config", _FakeService)

    now[0] = 20.0
    assert manager.find("service-0") is service
    now[0] = 21.0
    assert manager.expire_leases() == 1
    assert service.close_count == 1

    recovered = manager.register("service-0", "session-0", b"config", _FakeService)
    now[0] = 30.0
    assert manager.get_for_session("service-0", "session-0") is recovered
    now[0] = 39.0
    assert manager.expire_leases() == 0
    assert manager.items() == (("service-0", recovered),)


def test_identical_concurrent_registration_shares_factory_result() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    created = []

    def factory() -> _FakeService:
        service = _FakeService()
        created.append(service)
        factory_started.set()
        assert release_factory.wait(5), "Service factory was not released"
        return service

    manager = _create_manager()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
        assert factory_started.wait(5), "Service factory did not start"
        second_future = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
        release_factory.set()

        first_service = first_future.result(timeout=5)
        second_service = second_future.result(timeout=5)

    assert first_service is second_service
    assert created == [first_service]


def test_pending_registration_cannot_be_unregistered() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    def factory() -> _FakeService:
        factory_started.set()
        assert release_factory.wait(5), "Service factory was not released"
        return _FakeService()

    manager = _create_manager()
    with ThreadPoolExecutor(max_workers=1) as executor:
        registration = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
        assert factory_started.wait(5), "Service factory did not start"
        try:
            with pytest.raises(StaleSessionError, match="stale"):
                manager.unregister("service-0", "other-session")
            assert not manager.unregister("service-0", "session-0")
        finally:
            release_factory.set()

        service = registration.result(timeout=5)

    assert manager.get_for_session("service-0", "session-0") is service


def test_manager_close_terminates_shared_registration_wait() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    wait_started = threading.Event()
    created = []

    def factory() -> _FakeService:
        service = _FakeService()
        created.append(service)
        factory_started.set()
        assert release_factory.wait(5), "Service factory was not released"
        return service

    manager = _create_manager()
    with ThreadPoolExecutor(max_workers=2) as executor:
        creator = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
        assert factory_started.wait(5), "Service factory did not start"

        registration = manager._states["service-0"]
        wait_for_result = registration.future.result

        def observe_wait():
            wait_started.set()
            return wait_for_result()

        with patch.object(registration.future, "result", side_effect=observe_wait):
            waiter = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
            assert wait_started.wait(5), "Concurrent registration did not wait on the shared flight"

            manager.close()
            with pytest.raises(RuntimeError, match="lifecycle manager is closed"):
                waiter.result(timeout=5)

            release_factory.set()
            with pytest.raises(RuntimeError, match="lifecycle manager is closed"):
                creator.result(timeout=5)

    assert len(created) == 1
    assert created[0].close_count == 1


def test_registration_conflict_and_retired_session_are_rejected() -> None:
    manager = _create_manager()
    manager.register("service-0", "session-0", b"config", _FakeService)

    with pytest.raises(RegistrationConflictError, match="different configuration"):
        manager.register("service-0", "session-0", b"changed", _FakeService)

    assert manager.unregister("service-0", "session-0")
    with pytest.raises(StaleSessionError, match="retired"):
        manager.register("service-0", "session-0", b"config", _FakeService)


def test_expired_session_recovers_only_with_the_same_fingerprint() -> None:
    now = [0.0]
    manager = _create_manager(lambda: now[0])
    first_service = manager.register("service-0", "session-0", b"config", _FakeService)

    now[0] = 11.0
    assert manager.expire_leases() == 1
    assert first_service.close_count == 1

    with pytest.raises(RegistrationConflictError, match="different configuration"):
        manager.register("service-0", "session-0", b"changed", _FakeService)

    second_service = manager.register("service-0", "session-0", b"config", _FakeService)
    assert second_service is not first_service


def test_failed_recovery_registration_remains_recoverable() -> None:
    now = [0.0]
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "session-0", b"config", _FakeService)

    now[0] = 11.0
    manager.expire_leases()

    def fail_factory() -> _FakeService:
        raise RuntimeError("registration failed")

    with pytest.raises(RuntimeError, match="registration failed"):
        manager.register("service-0", "session-0", b"config", fail_factory)

    recovered = manager.register("service-0", "session-0", b"config", _FakeService)
    assert manager.get_for_session("service-0", "session-0") is recovered


def test_unregister_does_not_retire_a_recovering_registration() -> None:
    now = [0.0]
    factory_started = threading.Event()
    release_factory = threading.Event()
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "session-0", b"config", _FakeService)

    now[0] = 11.0
    manager.expire_leases()

    def factory() -> _FakeService:
        factory_started.set()
        assert release_factory.wait(5), "Recovery factory was not released"
        return _FakeService()

    with ThreadPoolExecutor(max_workers=1) as executor:
        recovery = executor.submit(manager.register, "service-0", "session-0", b"config", factory)
        assert factory_started.wait(5), "Recovery factory did not start"
        try:
            assert not manager.unregister("service-0", "session-0")
        finally:
            release_factory.set()

        recovered = recovery.result(timeout=5)

    assert manager.get_for_session("service-0", "session-0") is recovered


def test_expiration_does_not_retain_service_after_close_failure() -> None:
    now = [0.0]
    manager = _create_manager(lambda: now[0])
    failed_service = manager.register("service-0", "session-0", b"config", _FailingCloseService)

    now[0] = 11.0
    assert manager.expire_leases() == 1
    assert failed_service.close_count == 1
    assert manager.count == 0

    replacement = manager.register("service-0", "session-0", b"config", _FakeService)
    assert replacement is not failed_service
    manager.close()


def test_expiration_uses_the_owner_close_handler() -> None:
    now = [0.0]
    expired = []

    def expire_service(identity: str, service: _FakeService) -> None:
        expired.append(identity)
        service.close()

    manager = _create_manager(lambda: now[0], owner_close_handler=expire_service)
    service = manager.register("service-0", "session-0", b"config", _FakeService)
    now[0] = 11.0

    assert manager.expire_leases() == 1
    assert expired == ["service-0"]
    assert service.close_count == 1


def test_manager_close_uses_the_owner_close_handler() -> None:
    closed_identities = []

    def close_service(identity: str, service: _FakeService) -> None:
        closed_identities.append(identity)
        service.close()

    manager = _create_manager(owner_close_handler=close_service)
    service = manager.register("service-0", "session-0", b"config", _FakeService)

    manager.close()

    assert closed_identities == ["service-0"]
    assert service.close_count == 1


def test_new_session_after_expiration_retires_the_old_session() -> None:
    now = [0.0]
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "old-session", b"config", _FakeService)

    now[0] = 11.0
    manager.expire_leases()
    manager.register("service-0", "new-session", b"config", _FakeService)

    with pytest.raises(StaleSessionError, match="retired"):
        manager.register("service-0", "old-session", b"config", _FakeService)


def test_registration_is_busy_while_expired_service_is_closing() -> None:
    now = [0.0]
    close_started = threading.Event()
    release_close = threading.Event()
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "session-0", b"config", lambda: _BlockingCloseService(close_started, release_close))

    now[0] = 11.0
    with ThreadPoolExecutor(max_workers=1) as executor:
        expiration = executor.submit(manager.expire_leases)
        assert close_started.wait(5), "Expired service did not start closing"
        try:
            with pytest.raises(ServiceBusyError, match="being expired"):
                manager.register("service-0", "session-0", b"config", _FakeService)
        finally:
            release_close.set()

        assert expiration.result(timeout=5) == 1

    assert manager.register("service-0", "session-0", b"config", _FakeService) is not None


def test_unregistering_recoverable_session_retires_it() -> None:
    now = [0.0]
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "session-0", b"config", _FakeService)

    now[0] = 11.0
    manager.expire_leases()

    assert manager.unregister("service-0", "session-0")
    with pytest.raises(StaleSessionError, match="retired"):
        manager.register("service-0", "session-0", b"config", _FakeService)


def test_maintenance_loop_expires_services_and_close_is_idempotent() -> None:
    now = [0.0]
    closed = threading.Event()
    manager = _create_manager(lambda: now[0])
    manager.register("service-0", "session-0", b"config", lambda: _FakeService(closed))

    now[0] = 11.0
    manager.start_maintenance()
    manager.start_maintenance()
    try:
        assert closed.wait(1), "Lifecycle maintenance did not expire the service"
    finally:
        manager.close()
        manager.close()

    assert not manager.is_running


def test_maintenance_loop_survives_expiration_failure() -> None:
    maintenance_finished = threading.Event()
    attempts = 0
    manager = _create_manager()

    def expire_leases() -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("expiration failed")
        maintenance_finished.set()
        return 0

    with (
        patch.object(manager, "expire_leases", side_effect=expire_leases),
        patch(f"{LIFECYCLE_MODULE}.logger.exception") as log_exception,
    ):
        manager.start_maintenance()
        try:
            assert maintenance_finished.wait(1), "Lifecycle maintenance stopped after an expiration failure"
        finally:
            manager.close()

        log_exception.assert_called_once_with("%s service lifecycle maintenance failed", "Test")

    assert attempts >= 2


def test_stop_maintenance_can_signal_without_waiting() -> None:
    expiration_started = threading.Event()
    release_expiration = threading.Event()
    manager = _create_manager()

    def expire_leases() -> int:
        expiration_started.set()
        release_expiration.wait()
        return 0

    with patch.object(manager, "expire_leases", side_effect=expire_leases):
        manager.start_maintenance()
        assert expiration_started.wait(1), "Lifecycle maintenance did not start"
        try:
            manager.stop_maintenance(wait=False)
            assert manager.is_running
        finally:
            release_expiration.set()
            manager.stop_maintenance()

    assert not manager.is_running


@pytest.mark.parametrize(
    ("lease_timeout_s", "check_interval_s", "field_name"),
    [(0.0, 1.0, "lease_timeout_s"), (1.0, 0.0, "check_interval_s")],
)
def test_lifecycle_manager_rejects_non_positive_intervals(
    lease_timeout_s: float,
    check_interval_s: float,
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _create_manager(lease_timeout_s=lease_timeout_s, check_interval_s=check_interval_s)
