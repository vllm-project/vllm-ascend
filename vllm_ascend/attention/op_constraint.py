from abc import ABC, abstractmethod

import numpy as np
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def op_constraint_extra_kw_keys(prefix: str) -> tuple[str, str, str]:
    """Keys for AscendInputBatch.extra_kwargs / build_attn_metadata (**extra_kwargs)."""
    return (
        f"{prefix}_query_start_loc",
        f"{prefix}_query_start_loc_np",
        f"{prefix}_num_reqs_padded",
    )


class OpConstraint(ABC):

    @staticmethod
    @abstractmethod
    def padding(num_reqs: int,
                query_start_loc_np: np.ndarray,
                decode_query_len: int,
                batch_desc: BatchExecutionDescriptor):
        raise NotImplementedError
    
class FiaOpConstraint(OpConstraint):
    @staticmethod
    def padding(num_reqs: int,
                query_start_loc_np: np.ndarray,
                decode_query_len: int,
                batch_desc: BatchExecutionDescriptor):
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            return None

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        if num_tokens_padded == num_reqs_padded * decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1
        return num_reqs_padded, query_start_loc_np