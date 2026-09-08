# SPDX-License-Identifier: Apache-2.0
"""UNO configuration helpers without worker imports."""


def get_uno_tree_options(vllm_config) -> dict | None:
    config = vllm_config.speculative_config
    if config is None or config.method != "uno":
        return None
    options = (getattr(vllm_config, "additional_config", None) or {}).get("uno_tree")
    if options is None:
        return None
    if not isinstance(options, dict) or set(options) != {"draft_width", "candidate_top_k"}:
        raise ValueError("additional_config.uno_tree requires draft_width and candidate_top_k.")
    width, top_k = options["draft_width"], options["candidate_top_k"]
    if type(width) is not int or type(top_k) is not int or not 2 <= width <= 16 or not 2 <= top_k <= 128:
        raise ValueError("UNO tree requires draft_width in [2,16] and candidate_top_k in [2,128].")
    if not width <= config.num_speculative_tokens <= 32:
        raise ValueError("UNO tree num_speculative_tokens is its node budget, from draft_width to 32.")
    return options
