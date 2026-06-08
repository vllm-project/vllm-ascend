import argparse
import asyncio
import gc
import json
import os
import random
from typing import Any

import numpy as np
import torch
import torch_npu
from hstu_inference_ranking_utils import InferenceDataset, get_data_loader
from torch.autograd.profiler import record_function
from vllm import LLM
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)

os.environ.update({
    # "GLOO_SOCKET_IFNAME": "enp23s0f3",
    # "TP_SOCKET_IFNAME": "enp23s0f3",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",  # 全新进程启动方式
    # "ASCEND_LAUNCH_BLOCKING": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",  # 内存分配配置，单进程多卡
    "HCCL_INTRA_ROCE_ENABLE": "1",  # 使用PCIe环路进行多卡通信
})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


FIXED_SEED = 42
set_seed(FIXED_SEED)

MODEL_PATH = os.getenv("MODEL_PATH")
USER_2_PREFILL_PROMPT_DICT: dict[int, Any] = {}
USER_2_PREFILL_SAMPLING_PARAMS_DICT: dict[int, Any] = {}
USER_2_DECODE_PROMPT_DICT: dict[int, Any] = {}
USER_2_DECODE_SAMPLING_PARAMS_DICT: dict[int, Any] = {}
USER_2_PD_MERGE_PROMPT_DICT: dict[int, Any] = {}
USER_2_PD_MERGE_SAMPLING_PARAMS_DICT: dict[int, Any] = {}

torch._dynamo.config.cache_size_limit = 64


def clear_npu_cache():
    gc.collect()
    if torch.npu.is_available():
        torch.npu.synchronize()
        torch.npu.empty_cache()
        print("NPU cache cleared.")


def parse_params_and_deal_json() -> dict:
    parser = argparse.ArgumentParser(description="处理输入参数")
    parser.add_argument("--embedding_dim", required=False, help="embedding维度")
    parser.add_argument("--num_heads", required=False, help="注意力头数")
    parser.add_argument("--dim", required=False, help="注意力维度")
    parser.add_argument("--max_seq_len", required=True, help="最大序列长度")
    parser.add_argument("--max_batch_size", required=True, help="批大小")
    parser.add_argument("--use_random", required=True, help="是否使用随机数据以及模型")
    parser.add_argument("--aclgraph", required=True, help="是否采用ACLGraph成图推理")
    parser.add_argument("--candidate_num", required=True, help="候选集长度")
    parser.add_argument("--has_ffn", required=True, help="是否添加FFN")
    parser.add_argument("--max_vocab_size", required=True, help="最大词量")
    parser.add_argument("--concat_batch", required=True, help="是否融合为一个batch")
    parser.add_argument("--profiler", required=False, help="是否抓取profiler")
    parser.add_argument("--max_model_len", required=True, help="最大模型长度")
    parser.add_argument("--range", required=True, help="range范围")
    parser.add_argument("--graph_step", required=True, help="成图最大挡位差")
    parser.add_argument("--is_async", required=False, help="异步调度")
    parser.add_argument("--block_size", required=False, help="block大小")

    args = parser.parse_args()
    embedding_dim = int(args.embedding_dim)
    num_heads = int(args.num_heads)
    dim = int(args.dim)
    batch_size = int(args.max_batch_size)
    max_seq_len = int(args.max_seq_len)
    use_random_model = bool(int(args.use_random))
    aclgraph = bool(int(args.aclgraph))
    candidate_num = int(args.candidate_num)
    ffn = bool(int(args.has_ffn))
    max_vocab_size = int(args.max_vocab_size)
    concat_batch = bool(int(args.concat_batch))
    start_profile = bool(int(args.profiler))
    max_model_len = int(args.max_model_len)
    range_num = int(args.range)
    graph_step = int(args.graph_step)
    is_async = bool(int(args.is_async))
    block_size = int(args.block_size)

    def generate_seq_stages(max_seq, step=512):
        stages: list[int] = []
        power = 5
        while len(stages) < 2 or stages[-1] - stages[-2] < step:
            val = 2**power
            stages.append(val)
            power += 1

        stages[-1] = stages[-2] + step
        next_val = stages[-1] + step

        while next_val <= max_seq:
            stages.append(next_val)
            next_val += step

        if stages[-1] < max_seq:
            stages.append(max_seq)

        return stages

    graph_seq_stages = generate_seq_stages(max_model_len // batch_size,
                                           graph_step)

    # 1. 读取 JSON 文件
    assert MODEL_PATH is not None
    file_path = MODEL_PATH + '/config.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    use_random_model = bool(int(use_random_model))
    if not use_random_model:
        embedding_dim = 512
        num_heads = 4
        dim = 256
        batch_size = 4
        use_random_model = False
        max_vocab_size = 10000000
        max_model_len = 4096

    print("=================================")
    print(
        f"updating config.json: embedding_dim = {embedding_dim}, num_heads = {num_heads}, dim = {dim}, batch_size = {batch_size}, max_model_len = {max_model_len}, \
        max_seq_len: {max_seq_len}, use_random_model = {use_random_model}, aclgraph = {aclgraph}, max_vocab_size: {max_vocab_size}, concat_batch: {concat_batch}, start_profiler: {start_profile}, \
        range = {range_num}, graph_seq_stages: {graph_seq_stages}")

    data['max_seq_len'] = max_model_len
    data['num_hidden_layers'] = num_heads
    data['hidden_size'] = embedding_dim
    data['max_batch_size'] = batch_size
    data['kv_cache_config']['max_seq_len'] = max_model_len
    data['use_random_model'] = use_random_model
    data['max_num_candidates'] = candidate_num
    hstu_config = data['hstu_config']
    hstu_config['hidden_size'] = embedding_dim
    hstu_config['num_attention_heads'] = num_heads
    hstu_config['head_dim'] = dim
    hstu_config['has_ffn'] = ffn
    embedding_configs = data['task_config']['embedding_configs']

    for config in embedding_configs:
        config['dim'] = embedding_dim
        if not config['table_name'] == 'item':
            config['vocab_size'] = embedding_dim
        else:
            config['vocab_size'] = max_vocab_size

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print("config.json update finished!")
    print("=================================")
    return {
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "aclgraph": aclgraph,
        "is_use_random": use_random_model,
        "candidate_num": candidate_num,
        "max_vocab_size": max_vocab_size,
        "concat_batch": concat_batch,
        "embedding_dim": embedding_dim,
        "profiler": start_profile,
        "max_model_len": max_model_len,
        "vocab_size": max_vocab_size,
        "range": range_num,
        "is_async": is_async,
        'block_size': block_size,
    }


def initialize_llm_and_dataset(params):
    MODEL_PATH = os.getenv("MODEL_PATH")
    dataset_base_path = os.getenv("DATASET_PATH")
    assert MODEL_PATH is not None
    assert dataset_base_path is not None
    seq_logs_file = os.path.join(dataset_base_path, "processed_seqs.csv")
    batch_logs_file = os.path.join(dataset_base_path, "processed_batches.csv")

    item_feature_name = "video_id"
    action_feature_name = "action_weights"
    contextual_feature_names = [
        "user_id",
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ]

    is_async = params['is_async']

    if not is_async:
        llm = LLM(model=MODEL_PATH,
                  skip_tokenizer_init=True,
                  enforce_eager=not params['aclgraph'],
                  max_num_batched_tokens=20480,
                  enable_prefix_caching=False,
                  additional_config={
                      "graph_model_compile_config": {
                          "level": 1 if params['aclgraph'] else 0,
                          "use_aclgraph": params['aclgraph'],
                      },
                      "ascend_scheduler_config": {
                          "enabled": True,
                      },
                  },
                  kv_transfer_config=KVTransferConfig(
                      kv_connector="MooncakeUserConnectorStore",
                      kv_role="kv_both",
                      kv_port="20001",
                      kv_connector_extra_config={
                          "use_ascend_direct": True,
                          "use_layerwise": True,
                          "prefill": {
                              "dp_size": 1,
                              "tp_size": 1
                          },
                          "decode": {
                              "dp_size": 1,
                              "tp_size": 1
                          },
                          "lookup_rpc_port": "1",
                          "backend": os.getenv("KVCACHE_BACKEND")
                      }),
                  block_size=params['block_size'])
        async_llm = None
    else:
        llm = None

        GR_ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_PATH,
            enforce_eager=not params['aclgraph'],
            skip_tokenizer_init=True,
            max_num_batched_tokens=20480,
            enable_prefix_caching=False,
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeUserConnectorStore",
                kv_role="kv_both",
                kv_port="20001",
                kv_connector_extra_config={
                    "use_ascend_direct": True,
                    "use_layerwise": True,
                    "prefill": {
                        "dp_size": 1,
                        "tp_size": 1
                    },
                    "decode": {
                        "dp_size": 1,
                        "tp_size": 1
                    },
                    "lookup_rpc_port": "1",
                    "backend": os.getenv("KVCACHE_BACKEND")
                }),
            block_size=params['block_size'])

        async_llm = AsyncLLM.from_engine_args(GR_ENGINE_ARGS)

    batch_size = 1

    dataset = InferenceDataset(
        seq_logs_file=seq_logs_file,
        batch_logs_file=batch_logs_file,
        batch_size=batch_size,
        max_seqlen=params['max_seq_len'],
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=params['candidate_num'],
        item_vocab_size=params['max_vocab_size'],
        userid_name="user_id",
        date_name="date",
        sequence_endptr_name="interval_indptr",
        timestamp_names=["interval_end_ts"],
    )

    dataloader = get_data_loader(dataset=dataset)
    print("DataLoader ready.")

    return llm, async_llm, dataset, dataloader


def get_profiler():
    torch_profiler_trace_dir = os.getenv("VLLM_TORCH_PROFILER_DIR")
    print("==============================================")
    print("Profiling enabled. Traces will be saved to: ",
          torch_profiler_trace_dir)
    print("==============================================")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=True,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            torch_profiler_trace_dir))
    return profiler


def get_user_data(max_seq_len, dataloader_iter, user_id=None):
    selected_dates = []
    selected_dates_endptrs: list[int] = []
    selected_uids = []
    uids = None
    sum_cnt = 0
    while (batch := next(dataloader_iter, None)) and (
            uids == batch[0] or uids is None) and sum_cnt < max_seq_len:
        if user_id is not None and user_id != batch[0][0]:
            continue

        uids = batch[0]
        dates, seq_endptrs = batch[1:]
        if dates not in selected_dates:
            if len(selected_dates_endptrs) > 0:
                sum_cnt += selected_dates_endptrs[-1]
            selected_dates += dates
            selected_dates_endptrs += seq_endptrs
            selected_uids += uids
            if sum_cnt + seq_endptrs[0] >= max_seq_len:
                selected_dates.pop()
                selected_dates_endptrs.pop()
                selected_uids.pop()
                break
        else:
            sum_cnt += seq_endptrs[0]
            if sum_cnt >= max_seq_len:
                break
            sum_cnt -= seq_endptrs[0]
            selected_dates_endptrs.pop()
            selected_dates_endptrs += seq_endptrs
    return selected_dates, selected_dates_endptrs, selected_uids, uids, dates, seq_endptrs


def get_seq_user_data(max_seq_len,
                      candidate_num,
                      dataloader_iter,
                      vocab_size,
                      embedding_dim,
                      with_new_his=None,
                      new_his_len=None,
                      user_id=None):
    selected_dates = []
    selected_uids = []
    selected_video_id = []
    selected_action_weights = []
    candidate_item = []
    new_his_action_weights = []
    uids = None
    sum_cnt = 0
    sum_new_his_cnt = 0
    sum_candidate_cnt = 0
    print("============ Start load seq data =============")
    while (batch :=
           next(dataloader_iter, None)) and (uids == batch[0] or uids is None) and (
               sum_cnt < max_seq_len or sum_candidate_cnt < candidate_num):
        if user_id is not None and user_id != batch[0][0] and user_id > batch[
                0][0]:
            continue

        if user_id is not None and user_id < batch[0][0]:
            print(f" - User {user_id} has no enough data!!")
            break

        uids = batch[0]
        dates, video_id, action_weights = batch[1:]
        batch_len = len(video_id)
        if sum_cnt + batch_len < max_seq_len:
            selected_dates += dates
            selected_uids += uids
            selected_video_id += video_id
            selected_action_weights += action_weights
            sum_cnt += batch_len
        elif sum_cnt + batch_len >= max_seq_len and sum_cnt < max_seq_len:
            selected_dates += dates
            selected_uids += uids
            selected_video_id += video_id[:max_seq_len - sum_cnt]
            selected_action_weights += action_weights[:max_seq_len - sum_cnt]
            sum_cnt = max_seq_len
        else:
            start_pos = 0
            if with_new_his:
                if sum_new_his_cnt + batch_len < new_his_len:
                    new_his_action_weights += action_weights
                    candidate_item += video_id
                    sum_new_his_cnt += batch_len
                    continue
                else:
                    new_his_action_weights += action_weights[:new_his_len -
                                                             sum_new_his_cnt]
                    candidate_item += video_id[:new_his_len - sum_new_his_cnt]
                    sum_new_his_cnt = new_his_len
                    start_pos = new_his_len - sum_new_his_cnt

            if sum_candidate_cnt + batch_len - start_pos < candidate_num:
                candidate_item += video_id[start_pos:]
                sum_candidate_cnt = sum_candidate_cnt + batch_len - start_pos
            else:
                candidate_item += video_id[start_pos:candidate_num -
                                           sum_candidate_cnt]
                sum_candidate_cnt = candidate_num

    if len(candidate_item) < candidate_num:
        print(" - Has no candidate data! Use random data...")
        candidate_item = torch.randint(vocab_size, (candidate_num, )).tolist()

    if len(selected_video_id) < max_seq_len:
        print("- Has no enough history data! Use random data...")
        pad_len = max_seq_len - len(selected_video_id)
        random_item_ids = torch.randint(vocab_size, (pad_len, )).tolist()
        random_action_weights = torch.randint(embedding_dim,
                                              (pad_len, )).tolist()
        selected_video_id += random_item_ids
        selected_action_weights += random_action_weights

    return selected_dates, selected_uids, selected_video_id, selected_action_weights, candidate_item


def generate_request_by_uid(dataloader, dataloader_iter, dataset, user_ids):
    max_batches_to_process = params['batch_size']
    batches_processed = 0
    max_seq_len = params['max_seq_len']
    vocab_size = params['vocab_size']
    max_candidate_num = params['candidate_num']
    embedding_dim = params['embedding_dim']

    prefill_engine_prompt_list = []
    decode_engine_prompt_list = []
    pd_merged_prompt_list = []
    used_uids: list[int] = []
    prefill_sampling_params_list = []
    decode_sampling_params_list = []
    pd_merged_sampling_params_list = []
    print("============ Start load data =============")
    while batches_processed < max_batches_to_process:
        try:
            user_id = user_ids[batches_processed]
            if batches_processed > 0 and user_id < used_uids[-1]:
                dataloader_iter = iter(dataloader)
            if USER_2_PREFILL_PROMPT_DICT.get(user_id, None) is None:
                _, selected_uids, selected_video_id, selected_action_weights, candidate_item = get_seq_user_data(
                    max_seq_len=max_seq_len,
                    candidate_num=max_candidate_num,
                    dataloader_iter=dataloader_iter,
                    vocab_size=params['vocab_size'],
                    embedding_dim=params['embedding_dim'],
                    with_new_his=False,
                    new_his_len=None,
                    user_id=user_ids[batches_processed])
                # print(f" - generate_request_by_uid: uid={selected_uids}, selected_video_id={selected_video_id}, selected_action_weights={selected_action_weights}, candidate_item={candidate_item}")
                uid = torch.tensor([selected_uids[0]],
                                   dtype=torch.int64,
                                   device='npu')

                item_ids = torch.tensor(selected_video_id,
                                        dtype=torch.int64,
                                        device='npu')
                item_ids = item_ids % vocab_size
                action_weights = torch.tensor(selected_action_weights,
                                              dtype=torch.int64,
                                              device='npu')
                action_weights = action_weights % embedding_dim + vocab_size  # 合表偏移
                preporcessed_his = torch.stack((item_ids, action_weights))
                preporcessed_his = preporcessed_his.transpose(0, 1)
                preporcessed_his = preporcessed_his.flatten()

                candidate_item = torch.tensor(candidate_item,
                                              dtype=torch.int64,
                                              device='npu')
                candidate_item = candidate_item % vocab_size

                prefill_prompt_token_ids = preporcessed_his
                prefill_sampling_params = SamplingParams(temperature=0,
                                                         max_tokens=1,
                                                         prompt_logprobs=1,
                                                         extra_args={
                                                             "uid":
                                                             uid.tolist(),
                                                             "request_stage":
                                                             0,
                                                             "candidate_num":
                                                             [0],
                                                         })
                engine_prompt = TokensPrompt(
                    prompt_token_ids=prefill_prompt_token_ids.tolist())
                USER_2_PREFILL_PROMPT_DICT[user_id] = engine_prompt
                USER_2_PREFILL_SAMPLING_PARAMS_DICT[
                    user_id] = prefill_sampling_params

                # 现在是纯候选集，无新增的行为序列
                decode_prompt_token_ids = candidate_item
                decode_sampling_params = SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    prompt_logprobs=1,
                    extra_args={
                        "uid": uid.tolist(),
                        "request_stage": 1,
                        "candidate_num": [candidate_item.shape[0]]
                    })
                decode_engine_prompt = TokensPrompt(
                    prompt_token_ids=decode_prompt_token_ids.tolist())
                USER_2_DECODE_PROMPT_DICT[user_id] = decode_engine_prompt
                USER_2_DECODE_SAMPLING_PARAMS_DICT[
                    user_id] = decode_sampling_params

                pd_merged_prompt_token_ids = torch.cat(
                    [preporcessed_his, candidate_item], dim=0)
                pd_merged_sampling_params = SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    prompt_logprobs=1,
                    extra_args={
                        "uid": uid.tolist(),
                        "request_stage": 2,
                        "candidate_num": [candidate_item.shape[0]]
                    })
                pd_merged_prompt = TokensPrompt(
                    prompt_token_ids=pd_merged_prompt_token_ids.tolist())
                USER_2_PD_MERGE_PROMPT_DICT[user_id] = pd_merged_prompt
                USER_2_PD_MERGE_SAMPLING_PARAMS_DICT[
                    user_id] = pd_merged_sampling_params

            else:
                print(f" - User {user_id} prefill prompt hit!")
                engine_prompt = USER_2_PREFILL_PROMPT_DICT.get(user_id)
                prefill_sampling_params = USER_2_PREFILL_SAMPLING_PARAMS_DICT.get(
                    user_id)

                decode_engine_prompt = USER_2_DECODE_PROMPT_DICT.get(user_id)
                decode_sampling_params = USER_2_DECODE_SAMPLING_PARAMS_DICT.get(
                    user_id)

                pd_merged_prompt = USER_2_PD_MERGE_PROMPT_DICT.get(user_id)
                pd_merged_sampling_params = USER_2_PD_MERGE_SAMPLING_PARAMS_DICT.get(
                    user_id)

            batches_processed += 1
            print(f" - Processing batch #{batches_processed}")
            print(f" - User ids: {user_id}")
            # print(f" - Selected dates: {selected_dates_0}")
            # print(f" - Selected dates endptrs: {selected_dates_endptrs_0}")
            # print(f" - Start ptrs: {start_ptrs}")
            # print(f" - Concat batch: {concat_batch}")

            print(
                f"- Prefill batch {batches_processed} seq_len = {len(engine_prompt['prompt_token_ids'])}"
            )

            prefill_engine_prompt_list.append(engine_prompt)
            prefill_sampling_params_list.append(prefill_sampling_params)

            print(
                f"- Decode batch {batches_processed} seq_len = {len(decode_engine_prompt['prompt_token_ids'])}"
            )

            decode_engine_prompt_list.append(decode_engine_prompt)
            decode_sampling_params_list.append(decode_sampling_params)

            print(
                f"- PD merge batch {batches_processed} seq_len = {len(pd_merged_prompt['prompt_token_ids'])}"
            )

            pd_merged_prompt_list.append(pd_merged_prompt)
            pd_merged_sampling_params_list.append(pd_merged_sampling_params)

            used_uids.append(user_id)

        except StopIteration:
            break
    return prefill_engine_prompt_list, decode_engine_prompt_list, pd_merged_prompt_list, used_uids, prefill_sampling_params_list, decode_sampling_params_list, pd_merged_sampling_params_list


async def generate(
    engine: AsyncLLM,
    prompt_list,
    sampling_params,
    is_prefill,
    request_id,
):
    output = []
    async for out in engine.generate(prompt_list, sampling_params, request_id):
        if not is_prefill:
            output.append(deal_prompt_logprobs(out))
    return output


async def validate_output(params):
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    llm, async_llm, dataset, dataloader = initialize_llm_and_dataset(params)

    dataloader_iter = iter(dataloader)

    profile = params['profiler']

    batch_size = params['batch_size']

    range_cnt = params['range']
    is_async = params['is_async']

    with record_function("#### llm.ranking ###"):
        for i in range(range_cnt):
            user_ids = list(range(i * batch_size, (i + 1) * batch_size))

            print(f" - Start to deal user = {user_ids} 's request...")
            prefill_engine_prompt_list, decode_engine_prompt_list, pd_merged_engine_prompt_list, used_uids, prefill_sampling_params_list, decode_sampling_params_list, pd_merged_sampling_params_list = generate_request_by_uid(
                dataloader, dataloader_iter, dataset, user_ids)

            if profile and i == 0:
                llm.start_profile()

            print(
                f" - Calling llm.generate() for uids = {user_ids} (Prefill -> Decode -> PD merged)..."
            )

            try:
                if is_async:
                    tasks = []
                    tasks.append(
                        asyncio.create_task(
                            generate(async_llm, prefill_engine_prompt_list[0],
                                     prefill_sampling_params_list[0], True,
                                     user_ids[0])))
                    tasks.append(
                        asyncio.create_task(
                            generate(async_llm, decode_engine_prompt_list[0],
                                     decode_sampling_params_list[0], False,
                                     user_ids[0] + 10)))
                else:
                    with record_function("#### prefill ###"):
                        outputs = llm.generate(prefill_engine_prompt_list,
                                               prefill_sampling_params_list,
                                               use_tqdm=False)
                    with record_function("#### decode ###"):
                        outputs = llm.generate(decode_engine_prompt_list,
                                               decode_sampling_params_list,
                                               use_tqdm=False)
                    decode_output = deal_prompt_logprobs(outputs)
                    print(f" - Decode prompt_logprobs tensor: {decode_output}")
                    with record_function("#### pd_merged ###"):
                        outputs = llm.generate(pd_merged_engine_prompt_list,
                                               pd_merged_sampling_params_list,
                                               use_tqdm=False)
                    decode_output = deal_prompt_logprobs(outputs)
                    print(
                        f" - pd_merge prompt_logprobs tensor: {decode_output}")
            finally:
                if profile and i == 0:
                    llm.stop_profile()


def deal_prompt_logprobs(outputs):
    tensor_outputs = []
    for output in outputs:
        if isinstance(output.prompt_logprobs[1], torch.Tensor):
            res_tensor = output.prompt_logprobs[1]
            tensor_outputs.append(res_tensor.view(1, -1))
        else:
            logprobs_list = [
                list(logprob_dict.values())[0].logprob
                for logprob_dict in output.prompt_logprobs[1:]
            ]

            # 转为 PyTorch Tensor
            logprobs_tensor = torch.tensor(logprobs_list, dtype=torch.bfloat16)
            tensor_outputs.append(logprobs_tensor)
    return tensor_outputs


if __name__ == "__main__":
    clear_npu_cache()
    params = parse_params_and_deal_json()
    asyncio.run(validate_output(params))
