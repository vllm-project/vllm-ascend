from vllm_ascend.config.vllm_ascend import get_ascend_config


def lmhead_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.lmhead_tensor_parallel_size > 0


def embedding_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.embedding_tensor_parallel_size > 0


def oproj_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.oproj_tensor_parallel_size > 0


def mlp_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.mlp_tensor_parallel_size > 0
