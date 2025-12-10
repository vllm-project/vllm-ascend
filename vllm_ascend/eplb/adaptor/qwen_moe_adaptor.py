from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor


class QwenMoeAdaptor(VllmEplbAdaptor):

    def __init__(self, model, **args):
        super().__init__(model, **args)
        self.init_eplb_params()
        self.init_eplb_param_dict()
        self.init_expert_maps()

    def init_eplb_params(self):
        self.num_dense_layers = 0
        self.global_expert_num = self.model.config.num_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

    def model_register(self, model_config):
        super().model_register(self)
        config = model_config.hf_config
        self.model.num_moe_layers = config.num_hidden_layers