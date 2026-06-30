import torch
from torch import nn
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor


ori_init_parallel_drafting_params = SpecDecodeBaseProposer._init_parallel_drafting_params
def new_init_parallel_drafting_params(self):
    model_hf_config = self.draft_model_config.hf_config
    if hasattr(model_hf_config, "mask_token_id"):
        self.parallel_drafting_token_id = model_hf_config.mask_token_id
    else:
        ori_init_parallel_drafting_params(self)

SpecDecodeBaseProposer._init_parallel_drafting_params = new_init_parallel_drafting_params


class DSparkConfidenceHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "markov_rank", getattr(config, "dspark_markov_rank", 256)))
        self.proj = ReplicatedLinear(
            config.hidden_size + rank,
            1,
            bias=True,  # released dspark_qwen3_*_block7 ckpt has confidence_head.proj.bias
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden_states, markov_embeds], dim=-1)
        confidence, _ = self.proj(x.float())  # ReplicatedLinear returns (output, bias)
        return confidence.squeeze(-1)


class DSparkMarkovHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        rank = int(getattr(config, "markov_rank", getattr(config, "dspark_markov_rank", 256)))
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            rank,
            params_dtype=torch.float32,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.markov_w1(token_ids)
        logits = self.logits_processor(
            self.markov_w2,
            embeds.view(-1, embeds.shape[-1]).float(),
        )
        return logits.view(*embeds.shape[:-1], -1), embeds


ori_init = DFlashQwen3Model.__init__

def new_init(
    self,
    *,
    vllm_config: VllmConfig,
    start_layer_id: int = 0,
    prefix: str = "",
) -> None:
    hf_config = vllm_config.speculative_config.draft_model_config.hf_config
    if hasattr(hf_config, "markov_head_type"):
        if not hasattr(hf_config, "dflash_config") or hf_config.dflash_config is None:
            hf_config.dflash_config = {}
            hf_config.dflash_config["target_layer_ids"] = hf_config.target_layer_ids
    ori_init(
        self,
        vllm_config=vllm_config,
        start_layer_id=start_layer_id,
        prefix=prefix,
    )
    if hasattr(hf_config, "markov_head_type"):
        self.markov_head = DSparkMarkovHead(vllm_config, prefix=f"{prefix}.markov_head")
        self.confidence_head = DSparkConfidenceHead(vllm_config, prefix=f"{prefix}.confidence_head")

DFlashQwen3Model.__init__ = new_init