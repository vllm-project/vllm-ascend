import threading

import torch


class AscendContextManager:

    def __init__(self, vllm_config):
        self.additional_config = vllm_config.additional_config
        self.micro_batch_num = self.additional_config.micro_batch_num

        self.device_streams = {
            "micro_batch_1": torch.npu.Stream(),
            "micro_batch_2": torch.npu.Stream()
        }
        self.device_event = torch.npu.Event()
        self.host_event = threading.Event()

    def get_stream(self, stream_name):
        if stream_name in self.device_streams:
            return self.device_streams[stream_name]
        else:
            raise RuntimeError(f"unknow stream name : {stream_name}")

    def record_event(self):
        current_stream = torch.npu.current_stream()
        # record the launch task for another stream
        self.device_event.record(current_stream)
        # notify another thread for task issue
        self.host_event.set()

    def wait_event(self):
        # Should only run after the signal is received from others
        self.host_event.wait()
        self.host_event.clear()
        current_stream = torch.npu.current_stream()
        current_stream.wait_event(self.device_event)

    def get_stream_context(self, stream_name):
        return torch.npu.stream(self.get_stream(stream_name))


def init_ascend_global_context_manager():
    global Context
    Context = AscendContextManager()


def get_global_context_manager():
    global Context
    return Context
