from queue import Queue


class RecoveryContext:
    def __init__(self, exception: "Exception", fault_queue: "Queue", back_up, is_dummy_run=False):
        self.exception = exception
        self.fault_queue = fault_queue
        self.back_up = back_up
        self.is_dummy_run = is_dummy_run
