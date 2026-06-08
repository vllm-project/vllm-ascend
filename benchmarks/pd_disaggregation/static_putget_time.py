import argparse
import os
from pathlib import Path

import pandas as pd


def calculate_bandwidth(ms_dur, size):
    return size * 1000 / ms_dur / 1024


def static_buildin_method(path, func_names, size=None):
    res = {}
    df = pd.read_json(path)
    num_layers = 8
    for func_name in func_names:
        # full_func_name = f'built-in method {func_name} of PyCapsule object'
        full_func_name = f"{func_name}"
        selected_df = df[df["name"].str.contains(full_func_name)]
        # res[f'mean {func_name}(ms)'] = (selected_df['dur'].mean() / 1000)
        total_num = len(selected_df)
        if total_num == 0:
            continue
        res["mean put (ms)"] = (
            selected_df["dur"][: total_num // 2].sum() / num_layers / 1000
        )
        res["mean get (ms)"] = (
            selected_df["dur"][total_num // 2 :].sum() / num_layers / 1000
        )
        if size is not None:
            res["bandwidth put (GB/s)"] = calculate_bandwidth(
                res["mean put (ms)"], size
            )
            res["bandwidth get (GB/s)"] = calculate_bandwidth(
                res["mean get (ms)"], size
            )

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize Mooncake put/get bandwidth from profiler traces."
    )
    parser.add_argument(
        "--src-path",
        default=os.getenv("PROFILING_TRACE_DIR", "./profiling"),
        help="Directory that contains trace_view.json files.",
    )
    parser.add_argument(
        "--output-file",
        default="buildin_method.csv",
        help="CSV file name written under --src-path.",
    )
    args = parser.parse_args()

    src_path = Path(args.src_path)
    output_file_name = args.output_file
    # tar_get_func = ['batch_put_from_layers', 'batch_get_into_layers']
    tar_get_func = ["SDMA_SQE"]
    # tar_get_func = ['batch_put_from_multi_buffers', 'batch_get_into_multi_buffers']
    path_list = src_path.glob("**/trace_view.json")

    output_res = []
    for path in path_list:
        file_name = os.path.sep.join(str(path).split(os.path.sep)[-3:-2])
        size = int(file_name.split("_")[2][:-2])
        res = static_buildin_method(path, tar_get_func, size)
        res["size(mb)"] = int(size)
        res["block_size"] = int(file_name.split("_")[1][5:])
        res["backend"] = file_name.split("_")[0]
        res["file_name"] = file_name
        output_res.append(res)
    df = pd.DataFrame(output_res)
    df = df.sort_values(by=["size(mb)", "block_size"], ascending=True)
    df.to_csv(str(src_path / output_file_name))
