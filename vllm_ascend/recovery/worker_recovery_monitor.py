# TODO: implement worker recovery monitor
class RecoveryMonitor:
    def __init__(self, engine_index: int, worker_count: int) -> None:
        self.engine_index = engine_index
        self.worker_count = worker_count
        self.step_results = {}