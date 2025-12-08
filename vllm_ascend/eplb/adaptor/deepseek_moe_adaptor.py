from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor


class DeepSeekMoeAdaptor(VllmEplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(model, **args)
        self.init_eplb_params()
        self.init_eplb_param_dict()
        self.init_expert_maps()

    def init_eplb_params(self):
        self.num_dense_layers = self.model.config.first_k_dense_replace
        self.global_expert_num = self.model.config.n_routed_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers
