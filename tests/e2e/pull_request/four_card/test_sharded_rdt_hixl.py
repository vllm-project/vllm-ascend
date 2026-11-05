# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the upstream-shaped FSDP2 -> DP2/EP example on four NPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_sharded_rdt_hixl(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig
    from vllm.utils.network_utils import get_open_port

    model_path = tmp_path / "tiny-moe"
    model_path.mkdir()
    Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=1024,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        architectures=["Qwen3MoeForCausalLM"],
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    ).save_pretrained(model_path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"token{i}": i for i in range(4, 128)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    ).save_pretrained(model_path)
    example = Path(__file__).resolve().parents[4] / "examples/rl/rlhf_sharded_rdt_hixl_small_ep.py"
    env = dict(
        os.environ,
        RDT_MODEL=str(model_path),
        RDT_VLLM_PORT=str(get_open_port()),
        RDT_RESULT_PATH=str(tmp_path / "result.json"),
    )
    with (tmp_path / "example.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, "-u", str(example)], env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1800
        )
    assert result.returncode == 0, (tmp_path / "example.log").read_text()
    evidence = json.loads((tmp_path / "result.json").read_text())
    assert evidence["changed"] and evidence["stable"] and evidence["shutdown"]
