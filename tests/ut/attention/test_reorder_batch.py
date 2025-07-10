from dataclasses import dataclass
from typing import List, Dict
@dataclass
class InputBatch:
    req_ids: List[int]

@dataclass
class SchedulerOutput:
    num_scheduled_tokens: Dict
    scheduled_spec_decode_tokens: Dict

input_batch = InputBatch(
    req_ids=[0,1,2,3]
)

scheduler_output = SchedulerOutput(
    num_scheduled_tokens = {0: 2, 1: 1, 2: 3, 3: 1},
    scheduled_spec_decode_tokens = {
        0: [1],  # 2 - 1 = 1 → decode
        1: [],   # 1 - 0 = 1 → decode
        2: [1, 1],  # 3 - 2 = 1 → decode
        3: []    # 1 - 0 = 1 → decode
    }
)
def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            # For torch air graph mode we treat spec decoding as decode.
            if self.torchair_graph_enabled:
                if num_tokens - num_spec_tokens == 1:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens
            # For eager mode we treat spec decoding as chunked prefill.
            else:
                if num_tokens == 1:
                    decodes.append(i)
                    num_decode_tokens += num_tokens
                else:
                    prefills.append(i)
                    num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch


reorder_batch(input_batch, scheduler_output)