#!/usr/bin/env bash
# Compare upstream and Ascend slot-mapping kernels one case per msprof process.
#
# Usage:
#   bash benchmarks/ops/compare_slot_mapping_cases.sh \
#     /home/lingmutian/tmp/slot_mapping_compare_cases.json \
#     /home/lingmutian/profile_dir/slot_mapping_compare

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_INPUT_FILE="/home/lingmutian/tmp/slot_mapping_compare_cases.json"
readonly DEFAULT_PROFILE_ROOT="/home/lingmutian/profile_dir/slot_mapping_compare"
readonly KERNEL_NAME="_compute_slot_mappings_kernel"

input_file="${1:-${DEFAULT_INPUT_FILE}}"
profile_root="${2:-${DEFAULT_PROFILE_ROOT}}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f "${input_file}" ]]; then
    echo "Input file does not exist: ${input_file}" >&2
    exit 1
fi

mkdir -p "${profile_root}"
result_file="${profile_root}/task_duration_us.tsv"
summary_file="${profile_root}/summary.tsv"
printf 'case\timplementation\tgrid\ttask_duration_us\tmsprof_log\n' > "${result_file}"

mapfile -t case_names < <("${python_bin}" - "${input_file}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    print(case["name"])
PY
)

cd "${REPO_ROOT}"
for case_name in "${case_names[@]}"; do
    for implementation in upstream ascend; do
        if [[ "${implementation}" == "upstream" ]]; then
            wrapper="benchmarks/ops/slot_mapping_compare_wrappers.py:launch_upstream"
        else
            wrapper="benchmarks/ops/slot_mapping_compare_wrappers.py:launch_ascend"
        fi

        profile_dir="${profile_root}/${implementation}_${case_name}"
        log_file="${profile_root}/${implementation}_${case_name}.log"
        echo "Profiling ${implementation}/${case_name}"
        msprof op \
            --output="${profile_dir}" \
            --aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline \
            --kernel-name="${KERNEL_NAME}" \
            --application="${python_bin} benchmarks/ops/benchmark_wrapper.py \
                --input-file ${input_file} \
                --wrapper ${wrapper} \
                --mode wrapper \
                --kernel ${KERNEL_NAME} \
                --case-name ${case_name} \
                --device npu:0 \
                --warmup 20 \
                --profiling-rounds 100" 2>&1 | tee "${log_file}"

        task_duration_us="$(grep -E 'Task Duration\(us\):' "${log_file}" | tail -n 1 | awk '{print $3}')"
        grid="$("${python_bin}" - "${input_file}" "${case_name}" <<'PY'
import json
import sys

for case in json.load(open(sys.argv[1], encoding="utf-8")):
    if case["name"] == sys.argv[2]:
        print("x".join(map(str, case["grid"])))
        break
PY
)"
        if [[ -z "${task_duration_us}" ]]; then
            echo "Task Duration(us) missing for ${implementation}/${case_name}" >&2
            exit 1
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${case_name}" "${implementation}" "${grid}" "${task_duration_us}" "${log_file}" \
            >> "${result_file}"
    done
done

echo "Wrote ${result_file}"
awk -F '\t' '
    NR == 1 { next }
    $2 == "upstream" {
        upstream[$1] = $4
        grid[$1] = $3
        case_order[++num_cases] = $1
    }
    $2 == "ascend" { ascend[$1] = $4; grid[$1] = $3 }
    END {
        print "case\tgrid\tupstream_us\tascend_us\tspeedup"
        for (index = 1; index <= num_cases; index++) {
            case_name = case_order[index]
            if (case_name in ascend && ascend[case_name] != 0) {
                printf "%s\t%s\t%.6f\t%.6f\t%.2fx\n", \
                    case_name, grid[case_name], upstream[case_name], \
                    ascend[case_name], upstream[case_name] / ascend[case_name]
            }
        }
    }
' "${result_file}" > "${summary_file}"
echo "Wrote ${summary_file}"
