# mypy: ignore-errors
"""Register IQuestCoderForCausalLM as an alias of LlamaForCausalLM.

IQuest-Coder-V1 is a LLaMA-family model (GQA, ``head_dim=128``, standard LLaMA
weight naming, ``clip_qkv``/``sliding_window`` both disabled). Its ``config.json``
declares a custom architecture name ``IQuestCoderForCausalLM`` that is not part of
vLLM's built-in registry, so vLLM refuses to load it out of the box.

Because the weights, attention layout and all core operators are identical to
LLaMA, we alias the architecture to the existing ``LlamaForCausalLM``
implementation. After this patch the model can be served / evaluated **without**
``--hf-overrides``::

    vllm serve IQuestLab/IQuest-Coder-V1-40B-Instruct --tensor-parallel-size 2 \\
        --trust-remote-code

Note: the model still ships a custom ``configuration_iquestcoder`` config class and
an ``IQuestCoderTokenizer`` tokenizer class, so ``--trust-remote-code`` is still
required at load time.
"""

from vllm.logger import logger
from vllm.model_executor.models.registry import ModelRegistry


def _patch_iquestcoder_registry_alias() -> None:
    try:
        ModelRegistry.register_model(
            "IQuestCoderForCausalLM",
            "vllm.model_executor.models.llama:LlamaForCausalLM",
        )
        logger.info("Registered IQuestCoderForCausalLM -> LlamaForCausalLM alias.")
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("Failed to register IQuestCoderForCausalLM alias: %s", e)


_patch_iquestcoder_registry_alias()
