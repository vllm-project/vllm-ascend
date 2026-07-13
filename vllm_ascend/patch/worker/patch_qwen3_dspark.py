from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

ori_init_parallel_drafting_params = SpecDecodeBaseProposer._init_parallel_drafting_params


def new_init_parallel_drafting_params(self):
    spec_config = self.vllm_config.speculative_config
    model_hf_config = self.draft_model_config.hf_config
    if spec_config.method == "dspark" and hasattr(model_hf_config, "mask_token_id"):
        self.parallel_drafting_token_id = model_hf_config.mask_token_id
    else:
        ori_init_parallel_drafting_params(self)


SpecDecodeBaseProposer._init_parallel_drafting_params = new_init_parallel_drafting_params
