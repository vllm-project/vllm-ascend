from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.worker.v1.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)
AscendSpecDecodeBaseProposer._run_merged_draft = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._run_merged_draft
)
