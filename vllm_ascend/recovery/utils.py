from typing import Tuple


def get_engine_recovery_bind_address(
    engine_index: int,
) -> Tuple[str, str, str]:
    return (
        f"ipc:///tmp/engine_recovery_step_xpub_{engine_index}",
        f"ipc:///tmp/engine_recovery_fault_report_pull_{engine_index}",
        f"ipc:///tmp/engine_recovery_step_result_pull_{engine_index}",
    )
