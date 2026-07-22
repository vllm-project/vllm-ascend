import pandas as pd
import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

base_dir = os.path.dirname(os.path.abspath(__file__))
src_engine = os.path.join(base_dir, "engine_step.xlsx")
src_time = os.path.join(base_dir, "time_analysis.xlsx")
dst = os.path.join(base_dir, "trace_model.xlsx")
dst_avg = os.path.join(base_dir, "trace_model_avg.xlsx")

sheet_configs = [
    {"source_sheet": "engine_step", "target_sheet": "engine_step"},
    {"source_sheet": "decode_engine_step", "target_sheet": "decode_engine_step"},
]

avg_rows = []

with pd.ExcelWriter(dst, engine="openpyxl") as writer:
    for config in sheet_configs:
        source_sheet = config["source_sheet"]
        target_sheet = config["target_sheet"]

        df = pd.read_excel(src_engine, sheet_name=source_sheet)

        result = pd.DataFrame()
        result["node"] = df["node"]
        result["total_tokens"] = df["total_tokens"]
        result["engine step总耗时(ms)"] = df["execute time(ms)"]
        result["模型执行耗时(ms)"] = df["execute_model_cost_time(ms)"]
        result["前处理耗时(ms)"] = (df["execute_model_start_time"] - df["engine_step start"]) * 1000
        result["后处理耗时(ms)"] = (df["engine_step end"] - df["execute_model_end_time"]) * 1000

        avg_row = {"来源": target_sheet}
        for col in result.columns:
            if col != "node":
                avg_row[col] = result[col].mean()

        result.loc[len(result)] = {"node": "平均值", **{col: avg_row[col] for col in avg_row if col != "来源"}}

        result.to_excel(writer, sheet_name=target_sheet, index=False)
        print(f"[{target_sheet}] Done! Rows: {len(result)}")
        print(result.to_string())

        avg_rows.append(avg_row)

    df_ta = pd.read_excel(src_time, sheet_name="time_analysis")
    df_ta = df_ta[df_ta["RequestID"].notna()]

    time_cols = [
        "PD api server get request",
        "Get prefill engine request and start pickle",
        "Finish process request in prefill engine",
        "Prefill add waiting queue",
        "Start process request in prefill engine",
        "try to schedule in waiting queue",
        "success add to seq groups",
        "Start to dispatch decode request",
        "Add need pulling sequence",
        "Start pull kv",
        "Finish pull kv",
    ]
    for col in time_cols:
        df_ta[col] = pd.to_numeric(df_ta[col], errors="coerce")

    result_ta = pd.DataFrame()
    result_ta["request p_node"] = df_ta["P_NODE"]
    result_ta["request d_node"] = df_ta["D_NODE"]
    result_ta["tokenizer排队耗时(ms)"] = (df_ta["Get prefill engine request and start pickle"] - df_ta["PD api server get request"]) * 1000
    result_ta["tokenizer耗时(ms)"] = (df_ta["Finish process request in prefill engine"] - df_ta["Get prefill engine request and start pickle"]) * 1000
    result_ta["waiting队列耗时(ms)"] = (df_ta["Prefill add waiting queue"] - df_ta["Start process request in prefill engine"]) * 1000
    result_ta["schedule耗时(ms)"] = (df_ta["success add to seq groups"] - df_ta["try to schedule in waiting queue"]) * 1000
    result_ta["pull kv耗时(ms)"] = (df_ta["Finish pull kv"] - df_ta["Start pull kv"]) * 1000
    result_ta["pull kv排队耗时(ms)"] = (df_ta["Add need pulling sequence"] - df_ta["Start to dispatch decode request"]) * 1000

    avg_row_ta = {"来源": "time_analysis"}
    for col in result_ta.columns:
        if col not in ("request p_node", "request d_node"):
            avg_row_ta[col] = result_ta[col].mean()

    result_ta.loc[len(result_ta)] = {"request p_node": "平均值", "request d_node": "", **{col: avg_row_ta[col] for col in avg_row_ta if col != "来源"}}

    result_ta.to_excel(writer, sheet_name="time_analysis", index=False)
    print(f"[time_analysis] Done! Rows: {len(result_ta)}")
    print(result_ta.to_string())

    avg_rows.append(avg_row_ta)

print(f"\nAll sheets saved to {dst}")

avg_df = pd.DataFrame(avg_rows)
avg_df.set_index("来源", inplace=True)
avg_df.reset_index(inplace=True)
avg_df.to_excel(dst_avg, index=False)
print(f"\nAverages saved to {dst_avg}")
print(avg_df.to_string())