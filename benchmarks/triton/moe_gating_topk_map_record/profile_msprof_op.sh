#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Profile one operator invocation per process; msprof op handles its warmup.
set -euo pipefail

device="$1"
tokens="$2"
experts="$3"
top_k="$4"
scoring="$5"
script="$6"
eplb_source="$7"
atomic_source="$8"
candidate_source="$9"
output_root="${10}"

case_name="t${tokens}_e${experts}_k${top_k}_${scoring}"
case_dir="${output_root}/${case_name}"
mkdir -p "${case_dir}"

profile_one() {
    local implementation="$1"
    local kernel_name="$2"
    local source="$3"
    local label="$4"
    local output="${case_dir}/${label}"
    local -a candidate_args=()
    if [[ "${implementation}" == "candidate" ]]; then
        candidate_args=(--candidate "${source}")
    fi
    ASCEND_RT_VISIBLE_DEVICES="${device}" msprof op \
        --kernel-name="${kernel_name}" \
        --aic-metrics=PipeUtilization \
        --output="${output}" \
        python "${script}" \
        --task profile --t "${tokens}" --e "${experts}" --k "${top_k}" \
        --scoring "${scoring}" --profile-implementation "${implementation}" \
        --eplb-source "${eplb_source}" "${candidate_args[@]}" \
        > "${output}.log" 2>&1
    grep -E 'Op Name:|Task Duration\(us\):|Block Dim:|Device Id:|Current Freq:|Rated Freq:' \
        "${output}.log" > "${output}.summary"
    if ! grep -q 'Task Duration(us):' "${output}.summary"; then
        echo "Missing duration: ${output}.log" >&2
        return 1
    fi
    printf '%s\n' "${label}:" && cat "${output}.summary"
}

profile_one baseline MoeGatingTopK '' mainline_gating
profile_one baseline _map_to_physical_kernel '' mainline_mapping
profile_one baseline _record_expert_tokens_kernel '' mainline_record
profile_one candidate _moe_gating_topk_map_record_kernel "${atomic_source}" atomic_fused
profile_one candidate _moe_gating_topk_map_record_kernel "${candidate_source}" grid_routing
profile_one candidate _reduce_grid_records_kernel "${candidate_source}" grid_reduce
if [[ "$#" -ge 11 && -n "${11}" ]]; then
    profile_one candidate gating_top_k_map_and_record_kernel "${11}" pr17574_fused
fi
