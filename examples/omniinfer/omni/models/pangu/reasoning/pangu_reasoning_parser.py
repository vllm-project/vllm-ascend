# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerBase

from vllm.logger import init_logger
from vllm.reasoning import Qwen3ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("pangu")
class PanguReasoningParser(Qwen3ReasoningParser):
    """
    Reasoning parser for the Qwen3 model.

    The Pangu model uses [unused16]...[unused17] tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by [unused16] and [unused17] tokens from the model's
    output.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)
        self.think_start_token = "[unused16]"
        self.think_end_token = "[unused17]"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if (self.think_start_token_id is None
                or self.think_end_token_id is None):
            raise RuntimeError(
                "Pangu reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")
