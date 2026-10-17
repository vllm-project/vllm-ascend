"""Start coverage.py in spawn workers when COVERAGE_PROCESS_START is set.

``coverage run`` only traces the pytest process. EngineCore and NPU workers
started with multiprocessing spawn are fresh interpreters; they pick this
module up from PYTHONPATH and call process_startup() during site init.
Without COVERAGE_PROCESS_START this is a no-op.
"""

try:
    import coverage
except ImportError:
    pass
else:
    coverage.process_startup()
