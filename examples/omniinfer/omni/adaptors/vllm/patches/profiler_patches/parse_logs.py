# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import csv
import os
import re
import sys
from collections import defaultdict

import openpyxl
import pandas as pd


def parse_trace_logs(root_dir):
    pattern = r'<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([a-z0-9-]+)(?:; Role:(\S+))?'
    data_by_request = defaultdict(dict)
    request_role = defaultdict(dict)
    action_timestamps = {}
    engine_step_lines = []
    decode_engine_step_lines = []

    time_analysis_path = os.path.join(root_dir, "time_analysis.xlsx")
    engine_step_path = os.path.join(root_dir, "engine_step.xlsx")
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.log'):
                    log_file_path = os.path.join(dirpath, filename)
                    print(f"Processing log file: {log_file_path}")
                    try:
                        with open(log_file_path, 'r',
                                  encoding='latin1') as file:
                            for line in file:
                                # for engine step
                                if "profile: " in line:
                                    st_idx = line.find("profile:") + len(
                                        "profile: ")
                                    line = line[st_idx:]
                                    # if "prefill" in line:
                                    if not "[]" in line:
                                        engine_step_lines.append(line)
                                    else:
                                        decode_engine_step_lines.append(line)
                                    continue
                                # for time analysis
                                if "<<<Action" in line:
                                    st_idx = line.find("<<<Action")
                                    line = line[st_idx:]  # skip prefix if any
                                    match = re.match(pattern, line.strip())
                                    if match:
                                        action, timestamp, request_id, role = match.groups(
                                        )
                                        role, ip = role.split("_")
                                        action = action.strip()
                                        timestamp = float(timestamp)
                                        data_by_request[request_id][
                                            action] = timestamp
                                        request_role[request_id][role] = ip
                                        if action not in action_timestamps or timestamp < action_timestamps[
                                                action]:
                                            action_timestamps[
                                                action] = timestamp
                    except Exception as e:
                        print(f"Error reading {log_file_path}: {str(e)}")

        # process time analysis
        if data_by_request:
            # Sort actions by the earliest timestamp
            sorted_actions = sorted(action_timestamps.keys(),
                                    key=lambda x: action_timestamps[x])
            fieldnames = ['RequestID', 'P_NODE', "D_NODE"] + sorted_actions

            data = []
            for request_id, actions in data_by_request.items():
                row = {
                    'RequestID': request_id,
                    'P_NODE': request_role[request_id]["prefill"],
                    'D_NODE': request_role[request_id]["decode"]
                }
                # Add timestamps for each action, "-" for missing actions
                for action in sorted_actions:
                    row[action] = actions.get(action, '-')
                data.append(row)

            df = pd.DataFrame(data, columns=fieldnames)
            with pd.ExcelWriter(time_analysis_path,
                                engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='time_analysis', index=False)

                summary_data = {
                    'RequestID':
                    list(data_by_request.keys()),
                    'ActionCount':
                    [len(actions) for actions in data_by_request.values()]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

            print(
                f"Successfully parsed time analysis files. Check {time_analysis_path}."
            )

        else:
            print("No valid action record found in any log files.")

        # Process engine_step_lines
        engine_step_headers = [
            'node', 'engine_step start', 'engine_step end', 'execute time(ms)',
            'running_reqs_num_after_step', 'total_tokens',
            'waiting_reqs_num_after_step', 'reqs_ids', 'bs_tokens',
            'execute_model_start_time', 'execute_model_end_time',
            'execute_model_cost_time(ms)', 'kv_cache_usage', 'kv_blocks_num',
            'start_free_block_num', 'end_free_block_num', 'cost_blocks_num',
            'engine_core_str'
        ]
        with pd.ExcelWriter(engine_step_path, engine='openpyxl') as writer:
            if len(engine_step_lines) != 0:
                engine_data = []
                for line in engine_step_lines:
                    values = line.split("|")
                    values[-1] = values[-1].split("=")[-1]
                    row = dict(zip(engine_step_headers, values))
                    engine_data.append(row)

                df_engine = pd.DataFrame(engine_data,
                                         columns=engine_step_headers)
                df_engine.to_excel(writer,
                                   sheet_name='engine_step',
                                   index=False)

                print(
                    f"Successfully parsed engine step logs. Added 'engine_step' {engine_step_path}."
                )
            else:
                print("No valid engine step record found in log files.")

            if len(decode_engine_step_lines) != 0:
                decode_data = []
                for line in decode_engine_step_lines:
                    values = line.split("|")
                    values[-1] = values[-1].split("=")[-1]
                    row = dict(zip(engine_step_headers, values))
                    decode_data.append(row)

                df_decode = pd.DataFrame(decode_data,
                                         columns=engine_step_headers)
                df_decode['prefix'] = df_decode['node'] + '_' + df_decode[
                    'engine_core_str'].str.extract(r'(\d+)', expand=False)
                df_decode.to_excel(writer,
                                   sheet_name='decode_engine_step',
                                   index=False)

                print(
                    f"Successfully parsed decode engine step logs. "
                    f"Added 'decode_engine_step' sheet to {engine_step_path}.")

                # dump die load and die time
                selected_columns_for_die_load = [
                    'execute_model_start_time', 'total_tokens',
                    'running_reqs_num_after_step',
                    'waiting_reqs_num_after_step',
                    'execute_model_cost_time(ms)', 'start_free_block_num',
                    'cost_blocks_num'
                ]
                grouped = df_decode.groupby('prefix')
                wide_blocks = []

                for prefix, group in grouped:
                    group = group.reset_index(drop=True)
                    filtered = group[selected_columns_for_die_load].copy()

                    # Rename columns with prefix
                    filtered.columns = [
                        f"{prefix}_{col}" for col in filtered.columns
                    ]

                    # Reset index for alignment and add to list
                    wide_blocks.append(filtered.reset_index(drop=True))
                final_df = pd.concat(wide_blocks, axis=1)
                final_df.to_excel(writer,
                                  sheet_name='decode_die_load',
                                  index=False)
                print(f"Successfully parsed decode die load. "
                      f"Added 'decode_die_load' sheet to {engine_step_path}.")

            else:
                print("No valid decode engine step record found in log files.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Please input log directory. e.g.: python parse_logs.py path/to/all_pd_logs_direcotry"
        )
        exit()
    root_dir = sys.argv[1]
    parse_trace_logs(root_dir)
