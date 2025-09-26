import numpy as np
import torch
from vllm.utils import is_pin_memory_available

def move_cached_tensors(cached_tensor_list, src):
    num_reqs = len(src)
    applied = [False] * num_reqs
    last_idx = cached_tensor_list[0].shape[0] - 1
    for cached_tensor in cached_tensor_list:
        for i in range(num_reqs):
            cur = i
            while src[cur] != cur and src[cur] != -1 and not applied[cur]:
                applied[cur] = True
                if cur == i:
                    cached_tensor[last_idx] = cached_tensor[i]
                if src[cur] == i:
                    cached_tensor[cur] = cached_tensor[last_idx]
                else:
                    cached_tensor[cur] = cached_tensor[src[cur]]
                cur = src[cur]

class PenaltyCache:
    def __init__(self, num_req, vocab_size, device):
        self.cached_req_ids = None
        self.prompt_mask = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.bool,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.output_mask = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.bool,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.output_bin_counts = torch.zeros((num_req + 1, vocab_size),
                                        dtype=torch.int64,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.ones_cpu = torch.ones(vocab_size, dtype=torch.bool, device="cpu")
    
    def permute_cached_reqs(self, new_req_ids):
        if self.cached_req_ids == None:
            self.cached_req_ids = new_req_ids
            return
        num_reqs = len(new_req_ids)
        src = [-1] * num_reqs
        for i in range(len(self.cached_req_ids)):
            try:
                index = new_req_ids.index(self.cached_req_ids[i])
                src[index] = i
            except ValueError:
                pass
        move_cached_tensors([self.prompt_mask, self.output_mask, self.output_bin_counts], src)
        self.cached_req_ids = new_req_ids
    
    def prepare_new_reqs(self, scheduled_new_reqs):
        for i in range(len(scheduled_new_reqs)):
            try:
                index = self.cached_req_ids.index(scheduled_new_reqs[i].req_id)
            except ValueError:
                raise RuntimeError("penalty cache: a scheduled new req is not in req id list")
            self.output_bin_counts[index] = torch.zeros_like(self.output_bin_counts[index])
            self.output_mask[index] = torch.zeros_like(self.output_mask[index])
            prompt_mask_cpu = torch.zeros_like(self.prompt_mask[index], device='cpu')
            prompt_token_ids_tensor = torch.tensor(scheduled_new_reqs[i].prompt_token_ids, dtype=torch.int64)
            prompt_mask_cpu.scatter_(dim=0, index=prompt_token_ids_tensor, src=self.ones_cpu)
            self.prompt_mask[index] = prompt_mask_cpu.to(device=self.prompt_mask.device)
        
    def do_penalty_from_samplinng_metadata(self, input_batch) -> tuple[bool, bool, bool]:
        num_reqs = len(input_batch.req_ids)
        do_frequency_penalties = torch.any(input_batch.frequency_penalties_cpu_tensor[:num_reqs] != 0)
        do_presence_penalties = torch.any(input_batch.presence_penalties_cpu_tensor[:num_reqs] != 0)
        do_repetition_penalties = torch.any(input_batch.repetition_penalties_cpu_tensor[:num_reqs] != 1)
        return (do_frequency_penalties, do_presence_penalties, do_repetition_penalties)
    
    def update_do_penalty(self, sampling_metadata, input_batch):
        (self.do_frequency_penalties, self.do_presence_penalties, self.do_repetition_penalties) = self.do_penalty_from_samplinng_metadata(input_batch)
    
    def prepare_cache(self, scheduled_new_reqs, req_ids, sampling_metadata, input_batch):
        if sampling_metadata.no_penalties:
            self.do_penalties = False
            return
        self.do_penalties = True
        self.permute_cached_reqs(input_batch.req_ids)
        self.update_do_penalty(sampling_metadata, input_batch)
        self.prepare_new_reqs(scheduled_new_reqs)
    
    def save_token_ids(self, sampled_token_ids):
        if not self.do_penalties:
            return
        sampled_token_ids = sampled_token_ids.view(-1)
        self.output_bin_counts.scatter_add_(1, sampled_token_ids.unsqueeze(0).to(dtype=torch.int64),
            torch.ones_like(sampled_token_ids.unsqueeze(0), dtype=torch.int64))
        self.output_mask = self.output_bin_counts > 0
    
    def revert_rejected_tokens(self, accepted_mask, token_ids):
        if not self.do_penalties:
            return
        rejected_mask = -(~accepted_mask).to(dtype=torch.int64)
        token_ids = torch.clone(token_ids).masked_fill_(accepted_mask, 0).to(dtype=torch.int64)
        self.output_bin_counts.scatter_add_(1, token_ids, rejected_mask)
        self.output_mask = self.output_bin_counts > 0

class ProbCache:
    def __init__(self, num_req, num_tokens, topk, device):
        self.cached_req_ids = None
        self.topk_spec_token_ids = torch.zeros((num_req + 1, num_tokens, topk),
                                        dtype=torch.int32,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.topk_spec_token_probs = torch.zeros((num_req + 1, num_tokens, topk),
                                        dtype=torch.bfloat16,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
        self.computed = torch.zeros(num_req + 1,
                                        dtype=torch.bool,
                                        device=device,
                                        pin_memory=is_pin_memory_available())
    
    def permute_cached_reqs(self, new_req_ids):
        if self.cached_req_ids == None:
            self.cached_req_ids = new_req_ids
            return
        num_reqs = len(new_req_ids)
        src = [-1] * num_reqs
        for i in range(len(self.cached_req_ids)):
            try:
                index = new_req_ids.index(self.cached_req_ids[i])
                src[index] = i
            except ValueError:
                pass
        move_cached_tensors([self.topk_spec_token_ids, self.topk_spec_token_probs, self.computed], src)
        self.cached_req_ids = new_req_ids
    
    def prepare_new_reqs(self, scheduled_new_reqs):
        for i in range(len(scheduled_new_reqs)):
            try:
                index = self.cached_req_ids.index(scheduled_new_reqs[i].req_id)
            except ValueError:
                raise RuntimeError("penalty cache: a scheduled new req is not in req id list")
            self.topk_spec_token_ids[index] = torch.zeros_like(self.topk_spec_token_ids[index])
            self.topk_spec_token_probs[index] = torch.zeros_like(self.topk_spec_token_probs[index])
            self.computed[index] = False
    
    def update_sparse_rejection_sampler(self, topk_spec_token_ids, topk_spec_token_probs, idx):
        batch_size = topk_spec_token_ids.shape[0]
        self.topk_spec_token_ids[:batch_size, idx, :] = topk_spec_token_ids
        self.topk_spec_token_probs[:batch_size, idx, :] = topk_spec_token_probs
        self.computed[:batch_size] = True
    
    def prepare_cache(self, scheduled_new_reqs, req_ids, sampling_metadata, input_batch):
        self.permute_cached_reqs(input_batch.req_ids)
        self.prepare_new_reqs(scheduled_new_reqs)