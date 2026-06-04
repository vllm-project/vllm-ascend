import json
import os
import sys
from typing import Any

# ruff: noqa: E402
root_path = os.path.abspath(__file__)
root_path = os.path.sep.join(
    root_path.split(os.path.sep)[:-2] + ['fuxi-alpha'])
sys.path.append(root_path)

import torch
from fuxi_alpha_kuairand_demo import (get_seq_user_data,
                                      initialize_llm_and_dataset,
                                      parse_params_and_deal_json)
from tqdm import tqdm  # type: ignore[import-untyped]
from vllm.inputs import TokensPrompt

USER_2_PREFILL_PROMPT_DICT: dict[int, Any] = {}
USER_2_DECODE_PROMPT_DICT: dict[int, Any] = {}
USER_2_PD_MERGE_PROMPT_DICT: dict[int, Any] = {}
DELIMITER_CANDIDATE = torch.tensor([-1], dtype=torch.int64, device='cpu')


def generate_request_by_uid(dataloader, params, dataset, user_ids):
    batches_processed = 0
    max_seq_len = params['max_seq_len']
    vocab_size = params['vocab_size']
    dataloader_iter = iter(dataloader)
    max_candidate_num = params['candidate_num']

    prefill_engine_prompt_list = []
    decode_engine_prompt_list = []
    pd_merged_prompt_list = []
    used_uids: list[int] = []
    # while batches_processed < max_batches_to_process:
    for batches_processed in tqdm(range(len(user_ids))):
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
                                   device='cpu')

                item_ids = torch.tensor(selected_video_id,
                                        dtype=torch.int64,
                                        device='cpu')
                item_ids = item_ids % vocab_size
                action_weights = torch.tensor(selected_action_weights,
                                              dtype=torch.int64,
                                              device='cpu')
                action_weights = action_weights % params[
                    'embedding_dim'] + vocab_size  # 合表偏移
                preporcessed_his = torch.stack((item_ids, action_weights))
                preporcessed_his = preporcessed_his.transpose(0, 1)
                preporcessed_his = preporcessed_his.flatten()

                candidate_item = torch.tensor(candidate_item,
                                              dtype=torch.int64,
                                              device='cpu')
                candidate_item = candidate_item % vocab_size

                prefill_additional_info = {
                    "uid": [uid.item()],
                    "request_stage": 0,
                    "candidate_num": [0]
                }
                decode_additional_info = {
                    "uid": [uid.item()],
                    "request_stage": 1,
                    "candidate_num": [max_candidate_num]
                }
                pd_merged_additional_info = {
                    "uid": [uid.item()],
                    "request_stage": 2,
                    "candidate_num": [max_candidate_num]
                }

                _ = torch.cat([uid, preporcessed_his, DELIMITER_CANDIDATE],
                              dim=0)
                engine_prompt = TokensPrompt(
                    prompt_token_ids=preporcessed_his.tolist())
                engine_prompt[
                    "additional_information"] = prefill_additional_info
                USER_2_PREFILL_PROMPT_DICT[user_id] = engine_prompt

                # TODO 现在是纯候选集，无新增的行为序列
                _ = torch.cat([uid, DELIMITER_CANDIDATE, candidate_item],
                              dim=0)
                decode_engine_prompt = TokensPrompt(
                    prompt_token_ids=candidate_item.tolist())
                decode_engine_prompt[
                    "additional_information"] = decode_additional_info
                USER_2_DECODE_PROMPT_DICT[user_id] = decode_engine_prompt

                pd_merged_prompt_token_ids = torch.cat(
                    [preporcessed_his, candidate_item], dim=0)
                pd_merged_prompt = TokensPrompt(
                    prompt_token_ids=pd_merged_prompt_token_ids.tolist())
                pd_merged_prompt[
                    "additional_information"] = pd_merged_additional_info
                USER_2_PD_MERGE_PROMPT_DICT[user_id] = pd_merged_prompt

            else:
                # print(f" - User {user_id} prefill prompt hit!")
                engine_prompt = USER_2_PREFILL_PROMPT_DICT.get(user_id)
                decode_engine_prompt = USER_2_DECODE_PROMPT_DICT.get(user_id)
                pd_merged_prompt = USER_2_PD_MERGE_PROMPT_DICT.get(user_id)

            # print(f" - Processing batch #{batches_processed}")
            # print(f" - User ids: {user_id}")
            # print(f" - Selected dates: {selected_dates_0}")
            # print(f" - Selected dates endptrs: {selected_dates_endptrs_0}")
            # print(f" - Start ptrs: {start_ptrs}")
            # print(f" - Concat batch: {concat_batch}")

            # print(f"- Prefill batch {batches_processed} seq_len = {len(engine_prompt['prompt_token_ids'])}")

            prefill_engine_prompt_list.append(engine_prompt)

            # print(f"- Decode batch {batches_processed} seq_len = {len(decode_engine_prompt['prompt_token_ids'])}")

            decode_engine_prompt_list.append(decode_engine_prompt)

            # print(f"- PD merge batch {batches_processed} seq_len = {len(pd_merged_prompt['prompt_token_ids'])}")

            pd_merged_prompt_list.append(pd_merged_prompt)

            used_uids.append(user_id)

        except StopIteration:
            break
    return prefill_engine_prompt_list, decode_engine_prompt_list, pd_merged_prompt_list, used_uids


def output_datasets2json(nums, output_path, params):
    _, _, dataset, dataloader = initialize_llm_and_dataset(params)
    user_ids = list(range(0, nums))

    prefill_engine_prompt_list = []
    decode_engine_prompt_list = []
    pd_merged_engine_prompt_list = []
    prefill_engine_prompt_list, decode_engine_prompt_list, pd_merged_engine_prompt_list, _ = generate_request_by_uid(
        dataloader, params, dataset, user_ids)
    p_output_path = output_path.replace('.jsonl', '_p.jsonl')
    with open(p_output_path, 'w', encoding='utf-8') as f:
        for prompt in prefill_engine_prompt_list:
            line = json.dumps(prompt, ensure_ascii=False)
            f.write(line + '\n')
    print(f"Saved p prompts to: {p_output_path}")

    d_output_path = output_path.replace('.jsonl', '_d.jsonl')
    with open(d_output_path, 'w', encoding='utf-8') as f:
        for prompt in decode_engine_prompt_list:
            line = json.dumps(prompt, ensure_ascii=False)
            f.write(line + '\n')
    print(f"Saved d prompts to: {d_output_path}")

    pd_output_path = output_path.replace('.jsonl', '_pd_merged.jsonl')
    with open(pd_output_path, 'w', encoding='utf-8') as f:
        for pd_prompt in pd_merged_engine_prompt_list:
            pd_line = json.dumps(pd_prompt, ensure_ascii=False)
            f.write(pd_line + '\n')
    print(f"Saved pd prompts to: {pd_output_path}")


if __name__ == '__main__':
    generate_users_num = int(os.getenv("HSTU_PROMPT_USER_NUM", "1000"))
    base_path = os.getenv("HSTU_PROMPT_OUTPUT_DIR", "./data")
    file_name = f'hstu_prompts_{generate_users_num}.jsonl'
    output_path = os.path.join(base_path, file_name)
    os.makedirs(base_path, exist_ok=True)
    params = parse_params_and_deal_json()
    output_datasets2json(generate_users_num, output_path, params)
