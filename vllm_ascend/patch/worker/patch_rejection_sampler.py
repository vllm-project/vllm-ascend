import vllm.v1.sample.rejection_sampler as rs

from vllm_ascend.sample.rejection_sampler import apply_sampling_constraints, expand_batch_to_tokens, rejection_sample

# TODO: delete this patch after apply_sampling_constraints and rejection_sample
#   are extracted to as class func of RejectionSampler
rs.apply_sampling_constraints = apply_sampling_constraints
rs.rejection_sample = rejection_sample
rs.expand_batch_to_tokens = expand_batch_to_tokens
