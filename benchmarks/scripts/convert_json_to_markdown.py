import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tabulate import tabulate

CUR_PATH = Path(__file__).parent.resolve()
# latency results and the keys that will be printed into markdown
latency_results = []
latency_column_mapping = {
    "test_name": "Test name",
    "avg_latency": "Mean latency (ms)",
    "P50": "Median latency (ms)",
    "P99": "P99 latency (ms)",
}

# throughput tests and the keys that will be printed into markdown
throughput_results = []
throughput_results_column_mapping = {
    "test_name": "Test name",
    "num_requests": "Num of reqs",
    "total_num_tokens": "Total num of tokens",
    "elapsed_time": "Elapsed time (s)",
    "requests_per_second": "Tput (req/s)",
    "tokens_per_second": "Tput (tok/s)",
}

# serving results and the keys that will be printed into markdown
serving_results = []
serving_column_mapping = {
    "test_name": "Test name",
    "request_rate": "Request rate (req/s)",
    "request_throughput": "Tput (req/s)",
    "output_throughput": "Output Tput (tok/s)",
    "median_ttft_ms": "TTFT (ms)",
    "median_tpot_ms": "TPOT (ms)",
    "median_itl_ms": "ITL (ms)",
}


def read_markdown(file):
    if os.path.exists(file):
        with open(file) as f:
            return f.read() + "\n"
    else:
        return f"{file} not found.\n"


def results_to_json(latency, throughput, serving):
    return json.dumps({
        'latency': latency.to_dict(),
        'throughput': throughput.to_dict(),
        'serving': serving.to_dict()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the results of the benchmark tests.")
    parser.add_argument(
        "--results_folder",
        type=str,
        default="../results/",
        help="The folder where the benchmark results are stored.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="../results/",
        help="The folder where the benchmark results are stored.")
    parser.add_argument("--markdown_template",
                        type=str,
                        default="./perf_result_template.md",
                        help="The template file for the markdown report.")
    parser.add_argument("--tag",
                        default="main",
                        help="Tag to be used for release message.")
    parser.add_argument("--commit_id",
                        default="",
                        help="Commit ID to be used for release message.")

    args = parser.parse_args()
    results_folder = (CUR_PATH / args.results_folder).resolve()
    output_folder = (CUR_PATH / args.output_folder).resolve()
    markdown_template = (CUR_PATH / args.markdown_template).resolve()

    # collect results
    for test_file in results_folder.glob("*.json"):

        with open(test_file) as f:
            raw_result = json.loads(f.read())

        if "serving" in str(test_file):
            # this result is generated via `benchmark_serving.py`

            # update the test name of this result
            raw_result.update({"test_name": test_file.stem})

            # add the result to raw_result
            serving_results.append(raw_result)
            continue

        elif "latency" in f.name:
            # this result is generated via `benchmark_latency.py`

            # update the test name of this result
            raw_result.update({"test_name": test_file.stem})

            # get different percentiles
            for perc in [10, 25, 50, 75, 90, 99]:
                # Multiply 1000 to convert the time unit from s to ms
                raw_result.update(
                    {f"P{perc}": 1000 * raw_result["percentiles"][str(perc)]})
            raw_result["avg_latency"] = raw_result["avg_latency"] * 1000

            # add the result to raw_result
            latency_results.append(raw_result)
            continue

        elif "throughput" in f.name:
            # this result is generated via `benchmark_throughput.py`

            # update the test name of this result
            raw_result.update({"test_name": test_file.stem})

            # add the result to raw_result
            throughput_results.append(raw_result)
            continue

        print(f"Skipping {test_file}")
    serving_results.sort(key=lambda x: (len(x['test_name']), x['test_name']))

    latency_results = pd.DataFrame.from_dict(latency_results)
    serving_results = pd.DataFrame.from_dict(serving_results)
    throughput_results = pd.DataFrame.from_dict(throughput_results)

    raw_results_json = results_to_json(latency_results, throughput_results,
                                       serving_results)

    # remapping the key, for visualization purpose
    if not latency_results.empty:
        latency_results = latency_results[list(
            latency_column_mapping.keys())].rename(
                columns=latency_column_mapping)
    if not serving_results.empty:
        serving_results = serving_results[list(
            serving_column_mapping.keys())].rename(
                columns=serving_column_mapping)
    if not throughput_results.empty:
        throughput_results = throughput_results[list(
            throughput_results_column_mapping.keys())].rename(
                columns=throughput_results_column_mapping)

    processed_results_json = results_to_json(latency_results,
                                             throughput_results,
                                             serving_results)

    # get markdown tables
    latency_md_table = tabulate(latency_results,
                                headers='keys',
                                tablefmt='pipe',
                                showindex=False)
    serving_md_table = tabulate(serving_results,
                                headers='keys',
                                tablefmt='pipe',
                                showindex=False)
    throughput_md_table = tabulate(throughput_results,
                                   headers='keys',
                                   tablefmt='pipe',
                                   showindex=False)

    # document the result
    print(output_folder)
    with open(output_folder / "benchmark_results.md", "w") as f:

        results = read_markdown(markdown_template)
        results = results.format(
            latency_tests_markdown_table=latency_md_table,
            throughput_tests_markdown_table=throughput_md_table,
            serving_tests_markdown_table=serving_md_table,
            benchmarking_results_in_json_string=processed_results_json)
        f.write(results)
