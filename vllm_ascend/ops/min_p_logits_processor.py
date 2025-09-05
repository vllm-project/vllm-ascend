import torch
from vllm.config import get_current_vllm_config
from vllm.v1.sample.logits_processor import MinPLogitsProcessor

original_min_p_logits_processor_init_func = MinPLogitsProcessor.__init__


def min_p_logits_processor_init_func(self, *args, **kwargs):
    original_min_p_logits_processor_init_func(self, *args, **kwargs)

    vllm_config = get_current_vllm_config()
    decode_max_num_seqs = getattr(vllm_config.scheduler_config,
                                  'decode_max_num_seqs', 0)
    # reinit MinPLogitsProcessor if decode_max_num_seqs configured
    if decode_max_num_seqs != 0:
        device = args[1]
        is_pin_memory = args[2]
        max_num_reqs = max(vllm_config.scheduler_config.max_num_seqs,
                           decode_max_num_seqs)

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=is_pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            self.min_p_device = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device=device)
        else:
            self.min_p_device = self.min_p_cpu_tensor
        self.min_p = self.min_p_device[:0]


MinPLogitsProcessor.__init__ = min_p_logits_processor_init_func
