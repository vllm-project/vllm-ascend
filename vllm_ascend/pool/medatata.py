import torch


class PoolingStates:
    # NOTE: This should be removed after we drop support of vLLM v0.12.0
    def __init__(self):
        # for chunked prefill with ALL pooling
        self.hidden_states_cache: list[torch.Tensor] = []

    def clean(self):
        self.hidden_states_cache.clear()
