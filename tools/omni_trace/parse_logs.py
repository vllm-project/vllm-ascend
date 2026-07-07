# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# NOTE: This optional report parser writes .xlsx files through pandas/openpyxl.
# Install those packages in the log-analysis environment before running it.

import csv
import re
import os
from collections import defaultdict
import sys
import pandas as pd
import openpyxl
import traceback


def _collect_engine_core_timing(root_dir):
    """
    First pass: scan ALL .log files under root_dir and merge profile_mainmodel /
    profile_mtpmodel timing into engine_core_info keyed by engine_core_str.

    Decode `profile:` lines and scheduler main/mtp lines are often in different
    processes and therefore different log files; a single linear pass would
    see `profile:` before worker lines and leave main/mtp timing as 0.
    """
    engine_core_info = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".log"):
                continue
            log_file_path = os.path.join(dirpath, filename)
            try:
                with open(log_file_path, "r", encoding="latin1") as file:
                    for line in file:
                        if "profile_mainmodel:" in line:
                            _get_main_model_info(engine_core_info, line)
                        elif "profile_mtpmodel:" in line:
                            _get_mtp_model_info(engine_core_info, line)
            except Exception as e:
                print(f"Error reading {log_file_path} in timing pre-scan: {str(e)}")
                print(traceback.print_exc())
    return engine_core_info


def parse_trace_logs(root_dir):
    pattern = (
        r"<<<Action: (.*?); Timestamp:([\d.]+); RequestID:([^;]+)(?:; Role:(\S+))?"
    )
    data_by_request = defaultdict(dict)
    request_role = defaultdict(dict)
    action_timestamps = {}
    engine_step_lines = []
    decode_engine_step_lines = []
    # profile: (scheduler) and profile_mainmodel/profile_mtpmodel (workers) are often in
    # different log_pid_*.log files and may appear in any order in a single file — build
    # timing map in a first pass over ALL logs before joining decode rows.
    engine_core_info = _collect_engine_core_timing(root_dir)
    if engine_core_info:
        print(
            f"Pre-scan: merged main/mtp timing for {len(engine_core_info)} "
            "engine_core_str key(s) from profile_mainmodel / profile_mtpmodel."
        )

    time_analysis_path = os.path.join(root_dir, "time_analysis.xlsx")
    engine_step_path = os.path.join(root_dir, "engine_step.xlsx")
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                _get_step_line(
                    pattern,
                    data_by_request,
                    request_role,
                    action_timestamps,
                    engine_step_lines,
                    decode_engine_step_lines,
                    engine_core_info,
                    dirpath,
                    filename,
                )

        # process time analysis
        if data_by_request:
            df_final = _get_final_df(data_by_request, request_role)

            with pd.ExcelWriter(time_analysis_path, engine="openpyxl") as writer:
                df_final.to_excel(writer, sheet_name="time_analysis", index=False)
                summary_data = {
                    "RequestID": list(data_by_request.keys()),
                    "ActionCount": [
                        len(actions) for actions in data_by_request.values()
                    ],
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

            print(
                f"Successfully parsed time analysis files. Check {time_analysis_path}."
            )

        else:
            print("No valid action record found in any log files.")

        # Process engine_step_lines
        engine_step_headers = _get_engine_step_headers()
        with pd.ExcelWriter(engine_step_path, engine="openpyxl") as writer:
            # engine_step_sheet
            _engine_step_sheet(
                engine_step_lines, engine_step_path, writer, engine_step_headers
            )

            _decode_engine_step_sheet(
                decode_engine_step_lines, engine_step_path, writer, engine_step_headers
            )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(traceback.print_exc())


def _get_engine_step_headers():
    return [
        "node",
        "engine_step start",
        "engine_step end",
        "execute time(ms)",
        "running_reqs_num_after_step",
        "total_tokens",
        "waiting_reqs_num_after_step",
        "reqs_ids",
        "bs_tokens",
        "execute_model_start_time",
        "execute_model_end_time",
        "execute_model_cost_time(ms)",
        "kv_cache_usage",
        "kv_blocks_num",
        "start_free_block_num",
        "end_free_block_num",
        "cost_blocks_num",
        "engine_core_str",
    ]


def _get_final_df(data_by_request, request_role):
    action_map = _get_action_map()
    fieldnames = ["RequestID", "P_NODE", "D_NODE"] + list(action_map.keys())
    data = []
    for request_id, actions in data_by_request.items():
        decode = request_role[request_id].get("decode")
        prefill = request_role[request_id].get("prefill")
        if decode is None or prefill is None:
            print(
                f'request_id: {request_role[request_id].get("request_id")} decode or prefill is None'
            )
            continue
        row = {"RequestID": request_id, "P_NODE": prefill, "D_NODE": decode}
        # Add timestamps for each action, "-" for missing actions
        for action in action_map.keys():
            row[action] = actions.get(action, "-")
        data.append(row)

    df = pd.DataFrame(data, columns=fieldnames)
    # chinese_row
    chinese_row = {"RequestID": "", "P_NODE": "", "D_NODE": ""}
    chinese_row.update(action_map)
    df_cn = pd.DataFrame([chinese_row], columns=fieldnames)
    df_final = pd.concat([df.iloc[:0], df_cn, df.iloc[0:]], ignore_index=True)
    return df_final


def _get_step_line(
    pattern,
    data_by_request,
    request_role,
    action_timestamps,
    engine_step_lines,
    decode_engine_step_lines,
    engine_core_info,
    dirpath,
    filename,
):
    if filename.endswith(".log"):
        log_file_path = os.path.join(dirpath, filename)
        print(f"Processing log file: {log_file_path}")
        try:
            with open(log_file_path, "r", encoding="latin1") as file:
                for line in file:
                    # main model info
                    if "profile_mainmodel:" in line:
                        _get_main_model_info(engine_core_info, line)
                        continue
                    # mtp model info
                    if "profile_mtpmodel:" in line:
                        _get_mtp_model_info(engine_core_info, line)
                        continue
                    # for engine step
                    if "profile: " in line:
                        st_idx = line.find("profile:") + len("profile: ")
                        line = line[st_idx:]
                        # if "prefill" in line:
                        if not "[]" in line:
                            engine_step_lines.append(line)
                        else:
                            line = _set_decode_info(
                                decode_engine_step_lines, engine_core_info, line
                            )
                        continue
                    # for time analysis
                    if "<<<Action" in line:
                        st_idx = line.find("<<<Action")
                        line = line[st_idx:]  # skip prefix if any
                        match = re.match(pattern, line.strip())
                        if match:
                            action, timestamp, request_id, role = match.groups()
                            role, ip = role.split("_")
                            action = action.strip()
                            timestamp = float(timestamp)
                            # min value
                            if (
                                action not in data_by_request[request_id]
                                or timestamp < data_by_request[request_id][action]
                            ):
                                data_by_request[request_id][action] = timestamp
                            request_role[request_id][role] = ip
                            if (
                                action not in action_timestamps
                                or timestamp < action_timestamps[action]
                            ):
                                action_timestamps[action] = timestamp
        except Exception as e:
            print(f"Error reading {log_file_path}: {str(e)}")
            print(traceback.print_exc())


def _decode_engine_step_sheet(
    decode_engine_step_lines, engine_step_path, writer, engine_step_headers
):
    if len(decode_engine_step_lines) != 0:
        mtp_model_main_model_headers = _get_mtp_model_main_model_headers()
        decode_data = []
        decode_engine_step_headers = engine_step_headers + mtp_model_main_model_headers
        for line in decode_engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(decode_engine_step_headers, values))
            decode_data.append(row)

        df_decode = pd.DataFrame(decode_data, columns=decode_engine_step_headers)
        df_decode["prefix"] = (
            df_decode["node"]
            + "_"
            + df_decode["engine_core_str"].str.extract(r"(\d+)", expand=False)
        )
        df_decode.to_excel(writer, sheet_name="decode_engine_step", index=False)

        print(
            f"Successfully parsed decode engine step logs. "
            f"Added 'decode_engine_step' sheet to {engine_step_path}."
        )

        # dump die load and die time
        _decode_die_load_sheet(engine_step_path, writer, df_decode)

    else:
        print("No valid decode engine step record found in log files.")


def _get_mtp_model_main_model_headers():
    return [
        "main_model_start_time",
        "main_model_end_time",
        "execute_main_model_cost_time",
        "mtp_model_start_time",
        "mtp_model_end_time",
        "execute_mtp_model_cost_time",
    ]


def _engine_step_sheet(
    engine_step_lines, engine_step_path, writer, engine_step_headers
):
    if len(engine_step_lines) != 0:
        engine_data = []
        for line in engine_step_lines:
            values = line.split("|")
            values[-1] = values[-1].split("=")[-1]
            row = dict(zip(engine_step_headers, values))
            engine_data.append(row)

        df_engine = pd.DataFrame(engine_data, columns=engine_step_headers)
        df_engine.to_excel(writer, sheet_name="engine_step", index=False)

        print(
            f"Successfully parsed engine step logs. Added 'engine_step' {engine_step_path}."
        )
    else:
        print("No valid engine step record found in log files.")


def _decode_die_load_sheet(engine_step_path, writer, df_decode):
    decode_die_load_columns = _get_decode_die_load_columns()
    grouped = df_decode.groupby("prefix")
    wide_blocks = []

    for prefix, group in grouped:
        group = group.reset_index(drop=True)
        filtered = group[decode_die_load_columns].copy()

        # Rename columns with prefix
        filtered.columns = [f"{prefix}_{col}" for col in filtered.columns]

        # Reset index for alignment and add to list
        wide_blocks.append(filtered.reset_index(drop=True))
    final_df = pd.concat(wide_blocks, axis=1)
    final_df.to_excel(writer, sheet_name="decode_die_load", index=False)
    print(
        f"Successfully parsed decode die load. "
        f"Added 'decode_die_load' sheet to {engine_step_path}."
    )


def _get_decode_die_load_columns():
    return [
        "execute_model_start_time",
        "total_tokens",
        "running_reqs_num_after_step",
        "waiting_reqs_num_after_step",
        "execute_model_cost_time(ms)",
        "start_free_block_num",
        "cost_blocks_num",
    ]


def _get_action_map() -> dict:
    return {
        "PD api server get request": "prefill api server收到请求",
        "Get prefill engine request and start pickle": "触发engine处理请求",
        "Finish process request in prefill engine": "engine结束tokennizer",
        "Start process request in prefill engine": "engine准备开始处理输入请求",
        "Prefill add waiting queue": "prefill 请求添加到waiting队列",
        "try to schedule in waiting queue": "首次尝试加入running队列",
        "fail to add result of kv insufficient": "首次kv不足加入失败",
        "Prefill get new_blocks": "P侧申请完成KV",
        "success add to seq groups": "成功加入running队列",
        "Prefill start execute_model": "P开始execute model",
        "Prefill start execute main model": "P开始execute main model",
        "Prefill done execute main model": "P完成execute main model",
        "Prefill start execute mtp model": "P开始execute mtp model",
        "Prefill done execute mtp model": "P完成execute mtp model",
        "Prefill done execute_model": "P完成execute model",
        "Start to send output in prefill stage": "engine异步发送输出",
        "Client get prefill output": "client收到输出并入队",
        "Pop output queues": "client出队",
        "Finish prefill pickle and start response": "api server收到请求准备返回",
        "Enter decode to generate": "decode api server收到请求准备处理",
        "Start to dispatch decode request": "进入engine分发请求",
        "Add need pulling sequence": "添加到need pulling队列",
        "Start pull kv": "开始pull kv",
        "Finish pull kv": "结束pull kv",
        "Prefill free kv blocks": "P侧释放KV(和前后列时间戳可能存在时钟误差)",
        "Start append running sequece for decode": "pull kv结束添加到running队列",
        "Start to send output": "触发首个decode token执行",
        "First decode output token": "decoder返回第一个token",
        "Second decode output token": "decoder返回第二个token",
        "Third decode output token": "decoder返回第三个token",
        "Finish decode pickle and start response": "api server收到推理结果",
    }


def _set_decode_info(decode_engine_step_lines, engine_core_info, line):
    # Key must match _get_main_model_info / _get_mtp_model_info: last "|" field is
    # engine_core_str (f'{pid}-{execute_model_start_time}'). Regex search is unsafe
    # because reqs_ids / lists may contain digit-dash patterns.
    parts = line.rstrip().split("|")
    core_str = parts[-1].strip() if parts else ""
    if core_str:
        info = engine_core_info.get(
            core_str,
            {
                "main_model_start_time": 0.0,
                "main_model_end_time": 0.0,
                "execute_main_model_cost_time": 0.0,
                "mtp_model_start_time": 0.0,
                "mtp_model_end_time": 0.0,
                "execute_mtp_model_cost_time": 0.0,
            },
        )
        line = (
            line.strip()
            + f"|{info.get('main_model_start_time')}|{info.get('main_model_end_time')}|{info.get('execute_main_model_cost_time')}|{info.get('mtp_model_start_time')}|{info.get('mtp_model_end_time')}|{info.get('execute_mtp_model_cost_time')}\n"
        )
    decode_engine_step_lines.append(line)
    return line


def _get_mtp_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        mtp_start = float(parts[1])
        mtp_end = float(parts[2])
        mtp_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "mtp_model_start_time": mtp_start,
                "mtp_model_end_time": mtp_end,
                "execute_mtp_model_cost_time": mtp_cost,
            }
        )


def _get_main_model_info(engine_core_info, line):
    parts = line.split("|")
    if len(parts) >= 5:
        core_str = parts[-1].strip()
        main_start = float(parts[1])
        main_end = float(parts[2])
        main_cost = float(parts[3])
        if core_str not in engine_core_info:
            engine_core_info[core_str] = {}
        engine_core_info[core_str].update(
            {
                "main_model_start_time": main_start,
                "main_model_end_time": main_end,
                "execute_main_model_cost_time": main_cost,
            }
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please input log directory. e.g.: python parse_logs.py path/to/all_pd_logs_direcotry")
        exit()
    root_dir = sys.argv[1]
    parse_trace_logs(root_dir)
