import time
from datetime import datetime
import pytest

DURATION_THRESHOLD = 120  
SLOW_COUNT_LIMIT = 5     


_per_file_slow_cases = {}
_current_file = None


def pytest_runtest_setup(item):
    item.start_time = time.time()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    
    global _current_file

    if report.when != 'call':
        return
    if not hasattr(item, 'start_time'):
        return
    time_stamp = datetime.now().strftime("[%H:%M:%S]")
    print(f"{time_stamp}")

    file_path = item.fspath
    duration = time.time() - item.start_time

    if file_path not in _per_file_slow_cases:
        _per_file_slow_cases[file_path] = 0

    if duration > DURATION_THRESHOLD:
        _per_file_slow_cases[file_path] += 1
        cnt = _per_file_slow_cases[file_path]
        print(f" Detected slow case ({cnt}/{SLOW_COUNT_LIMIT}): {duration:.2f}s | {item.nodeid}")

        if cnt >= SLOW_COUNT_LIMIT:
            print(f"\n Timeout cases in {file_path} ≥ {SLOW_COUNT_LIMIT}\n")
            _current_file = file_path


def pytest_runtest_call(item):
    if _current_file == item.fspath:
        print(f"CASE SKIP: {item.nodeid}")
        pytest.skip("The use case takes too long.")