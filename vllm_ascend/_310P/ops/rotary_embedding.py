from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding


class AscendMRotaryEmbedding310(MRotaryEmbedding):

    def forward_oot(self, positions, query, key):
        return super().forward_oot(positions, query, key)
