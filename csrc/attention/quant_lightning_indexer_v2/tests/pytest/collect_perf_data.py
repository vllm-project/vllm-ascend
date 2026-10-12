#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import glob
import os

import pandas as pd


def extract_qliv2_row(prof_folder):
    prof_output = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.isdir(prof_output):
        return None

    csv_files = glob.glob(os.path.join(prof_output, "op_summary*.csv"))
    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])
    target = df[df["Op Name"] == "QuantLightningIndexerV2"]
    if target.empty:
        return None

    row = target.iloc[0].to_dict()
    return row


def collect_and_save_incremental(prof_folder, test_result_path):
    """增量模式：读取单条用例的性能数据，追加到 result_perf.xlsx"""
    if not os.path.exists(test_result_path):
        print(f"结果文件不存在: {test_result_path}")
        return None

    df_result = pd.read_excel(test_result_path)
    if df_result.empty:
        return None

    row_data = extract_qliv2_row(prof_folder)
    if row_data is None:
        return None

    last_idx = df_result.shape[0] - 1
    case_name = df_result.iloc[last_idx]["case_name"]

    perf_dict = {last_idx: row_data}
    perf_df = pd.DataFrame.from_dict(perf_dict, orient="index")

    overlap = set(df_result.columns) & set(perf_df.columns)
    if overlap:
        rename_map = {col: f"op_{col}" for col in overlap}
        perf_df = perf_df.rename(columns=rename_map)

    perf_path = test_result_path.replace(".xlsx", "_perf.xlsx")
    tmp_path = perf_path + ".tmp.xlsx"

    if os.path.exists(perf_path):
        df_perf = pd.read_excel(perf_path)
        # 以 df_result 为基准重建，保留已有 perf 列，追加新行
        perf_cols = [c for c in df_perf.columns if c not in df_result.columns]
        df_merged = df_result.copy()
        for col in perf_cols:
            df_merged[col] = None
            for idx in range(min(len(df_result), len(df_perf))):
                if col in df_perf.columns and idx < len(df_perf):
                    val = df_perf.at[idx, col]
                    if not (isinstance(val, float) and pd.isna(val)):
                        df_merged.at[idx, col] = val
        for col in perf_df.columns:
            if col not in df_merged.columns:
                df_merged[col] = None
            df_merged.loc[last_idx, col] = perf_df.loc[last_idx, col]
        df_merged.to_excel(tmp_path, index=False)
        os.replace(tmp_path, perf_path)
    else:
        df_result_with_perf = pd.concat([df_result, perf_df], axis=1)
        df_result_with_perf.to_excel(tmp_path, index=False)
        os.replace(tmp_path, perf_path)

    print(f"  [perf] {case_name} Task Duration: {row_data['Task Duration(us)']}us -> {perf_path}")
    return row_data


def collect_all(test_result_path, is_compare=False, perf_golden_path="perf_golden.xlsx"):
    """批量模式：收集所有 PROF 文件夹的性能数据"""
    if not os.path.exists(test_result_path):
        print(f"结果文件不存在: {test_result_path}")
        return

    df_b = pd.read_excel(test_result_path)
    valid_mask = df_b["result"] != "NPU ERROR"
    valid_count = valid_mask.sum()

    if valid_count == 0:
        print("没有有效用例,跳过性能数据收集")
        return

    prof_folders = sorted(
        [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("PROF")], key=lambda x: os.path.getmtime(x)
    )

    print("============= 开始收集性能数据 =============")
    print(f"有效用例数: {valid_count}, PROF文件夹数: {len(prof_folders)}")

    if len(prof_folders) == 0:
        print("未找到PROF文件夹, 跳过性能数据收集")
        return

    if len(prof_folders) != valid_count:
        print(f"警告: PROF文件夹数量({len(prof_folders)})与有效用例数({valid_count})不一致")

    perf_rows = {}
    prof_idx = 0

    for i in range(df_b.shape[0]):
        if not valid_mask.iloc[i]:
            continue

        if prof_idx >= len(prof_folders):
            print(f"  [{i}] PROF文件夹不足, 跳过剩余用例")
            break

        prof = prof_folders[prof_idx]
        case_name = df_b.iloc[i]["case_name"]
        row_data = extract_qliv2_row(prof)

        if row_data is not None:
            perf_rows[i] = row_data
            print(f"  [{prof_idx + 1}] {case_name} -> {prof} (Task Duration: {row_data['Task Duration(us)']}us)")
        else:
            print(f"  [{prof_idx + 1}] {case_name}: 未找到QuantLightningIndexer数据")

        prof_idx += 1

    if not perf_rows:
        print("未收集到任何性能数据")
        return

    perf_df = pd.DataFrame.from_dict(perf_rows, orient="index")
    overlap = set(df_b.columns) & set(perf_df.columns)
    if overlap:
        rename_map = {col: f"op_{col}" for col in overlap}
        perf_df = perf_df.rename(columns=rename_map)
        print(f"op_summary列名冲突已重命名: {list(rename_map.values())}")

    df_b = pd.concat([df_b, perf_df], axis=1)

    if is_compare:
        try:
            df_c = pd.read_excel(perf_golden_path)
        except Exception as e:
            print(f"读取基线数据失败: {e}")
            is_compare = False

    if is_compare:
        perf_threshold = 10
        perf_fail_list = []
        df_b["qliv2_perf_diff"] = ""
        df_b["perf_result"] = ""

        for i in perf_rows:
            try:
                cur_sas = df_b.at[i, "Task Duration(us)"]
                golden_sas = df_c.iloc[i]["Task Duration(us)"]
                sas_diff = float(cur_sas) - float(golden_sas)
                df_b.at[i, "qliv2_perf_diff"] = sas_diff

                if abs(sas_diff) > perf_threshold:
                    df_b.at[i, "perf_result"] = "Failed"
                    perf_fail_list.append(df_b.iloc[i]["case_name"])
                else:
                    df_b.at[i, "perf_result"] = "Pass"
            except Exception as e:
                print(f"  基线对比出错 (行{i}): {e}")

        if perf_fail_list:
            print(f"性能不达标用例: {perf_fail_list}")

    new_path = test_result_path.replace(".xlsx", "_perf.xlsx")
    tmp_path = new_path + ".tmp.xlsx"
    if os.path.exists(new_path):
        existing = pd.read_excel(new_path)
        existing = existing.reindex(range(df_b.shape[0]))
        for col in df_b.columns:
            for idx in range(df_b.shape[0]):
                if pd.isna(existing.at[idx, col]):
                    existing.at[idx, col] = df_b.at[idx, col]
        op_cols = [c for c in existing.columns if c.startswith("op_") or c == "Task Duration(us)"]
        filled_count = existing[op_cols].dropna(how="all").shape[0] if op_cols else 0
        if filled_count >= len(perf_rows):
            print(f"增量模式已收集 {filled_count}/{len(perf_rows)} 条性能数据，跳过批量补采")
            return
        print(f"增量模式仅 {filled_count}/{len(perf_rows)} 条，批量补采覆盖")
        for col in perf_df.columns:
            if col in existing.columns:
                for idx in perf_rows:
                    if pd.isna(existing.at[idx, col]):
                        existing.at[idx, col] = perf_df.at[idx, col]
            else:
                existing[col] = None
                for idx in perf_rows:
                    existing.at[idx, col] = perf_df.at[idx, col]
        existing.to_excel(tmp_path, index=False)
        os.replace(tmp_path, new_path)
        return

    df_b.to_excel(tmp_path, index=False)
    os.replace(tmp_path, new_path)

    print(f"\n性能数据已保存: {new_path}")
    print(f"共拼接 {len(perf_rows)} 条用例的 op_summary 全字段数据")
    print("============= 性能数据收集完成 =============")


def main():
    parser = argparse.ArgumentParser(description="收集性能数据")
    parser.add_argument("--test_result_path", type=str, default="result.xlsx")
    parser.add_argument(
        "--incremental", action="store_true", default=False, help="增量模式：只收集最新一条用例的性能数据"
    )
    parser.add_argument("--prof_folder", type=str, default=None, help="增量模式下指定 PROF 文件夹路径")
    parser.add_argument("--is_compare", action="store_true", default=False)
    parser.add_argument("--perf_golden_path", type=str, default="perf_golden.xlsx")
    args = parser.parse_args()

    if args.incremental:
        if args.prof_folder:
            prof = args.prof_folder
        else:
            prof_folders = sorted(
                [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith("PROF")],
                key=lambda x: os.path.getmtime(x),
            )
            if not prof_folders:
                print("未找到PROF文件夹")
                return
            prof = prof_folders[-1]

        collect_and_save_incremental(prof, args.test_result_path)
    else:
        collect_all(args.test_result_path, args.is_compare, args.perf_golden_path)


if __name__ == "__main__":
    main()
