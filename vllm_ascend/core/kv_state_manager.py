from vllm.v1.request import Request


class KVStateManager:

    def __init__(
        self,
        max_num_seqs: int,
    ):
        self.max_num_seqs = max_num_seqs
        # same as kv_cache, keep state 0 empty for padding in block_table
        self.states_pool: set[int] = set(range(1, max_num_seqs + 1))
        self.req_to_state_id: dict[str, int] = {}

    def allocate_slots(
        self,
        request: Request = None,
    ) -> int | None:
        if len(self.states_pool) == 0:
            return None
        request_id = request.request_id
        if request_id in self.req_to_state_id:
            raise ValueError(
                f"Request {request_id} already allocated, only allocate state while prefill."
            )
        state_to_allocate = self.states_pool.pop()
        self.req_to_state_id[request_id] = state_to_allocate
        self.get_resource_usage()
        return state_to_allocate

    def free(
        self,
        request: Request = None,
    ) -> None:
        request_id = request.request_id
        if request_id not in self.req_to_state_id:
            # NOTE(zxr): sometimes, a request may not allocate state but free, it should not raise an error
            # raise ValueError(f"Request {request_id} not allocated a state, unable to free.")
            return
        state_to_free = self.req_to_state_id.pop(request_id)
        self.states_pool.add(state_to_free)

    def get_resource_usage(self) -> tuple[int, int, int]:
        # For monitoring usage, return
        # (in use state nums, remained state nums, total state nums)
        state_num_remain = len(self.states_pool)
        state_num_in_use = self.max_num_seqs - state_num_remain
        return (state_num_in_use, state_num_remain, self.max_num_seqs)
