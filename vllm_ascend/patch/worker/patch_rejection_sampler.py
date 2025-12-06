import vllm.v1.sample.rejection_sampler as rs

from vllm_ascend.sample.rejection_sampler import (expand_batch_to_tokens,
                                                  rejection_sample)

rs.expand_batch_to_tokens = expand_batch_to_tokens
rs.rejection_sample = rejection_sample