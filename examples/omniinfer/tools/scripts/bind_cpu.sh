#!/bin/bash
set -e

MARKER="/tmp/process/proc_trace.txt"
LOG_FILE="/tmp/process/bind_cpu.log"

ROLE=${ROLE:-P}
[[ "$ROLE" == "P" || "$ROLE" == "D" ]] || { echo "ROLE must be P or D" >&2; exit 1; }

mkdir -p "$(dirname "$LOG_FILE")" && >"$LOG_FILE"

# check total number of CPUs
max_cpu=$(lscpu -p=CPU | grep -v '^#' | awk 'BEGIN{max=0} {if($1>max)max=$1} END{print max}')
if [[ $max_cpu -ge 319 ]]; then
    total_cpu=320
elif [[ $max_cpu -ge 179 ]]; then
    total_cpu=180
else
    total_cpu=$((max_cpu+1))
fi
free_cpus=($(seq 0 $((total_cpu-1))))

# remove 0,20,40,..., suit for CPU_AFFINITY_MODE=2
for ((i=0; i<total_cpu; i+=20)); do
    for idx in "${!free_cpus[@]}"; do
        [[ ${free_cpus[$idx]} -eq $i ]] && unset 'free_cpus[idx]'
    done
done
free_cpus=("${free_cpus[@]}") # updata free cpus

declare -A idx_map
for i in "${!free_cpus[@]}"; do idx_map[${free_cpus[i]}]=$i; done

declare -A keep_pids
while IFS='|' read -r pid _ _; do
    [[ -n $pid ]] && keep_pids[$pid]=1
done < <(
    awk -F'[[:space:]]+' '{
        pid=tag=lr=""
        for(i=1;i<=NF;i++){
            if($i ~ /^pid=/)        pid=substr($i,5)
            if($i ~ /^tag=/)        tag=substr($i,5)
            if($i ~ /^local_rank=/) lr=substr($i,12)
        }
        print pid "|" tag "|" lr
    }' "$MARKER"
)

while read -r cpu; do
    [[ -n ${idx_map[$cpu]} ]] || continue          
    idx=${idx_map[$cpu]}
    last_idx=$((${#free_cpus[@]}-1))
    last_cpu=${free_cpus[$last_idx]}
    free_cpus[$idx]=$last_cpu
    idx_map[$last_cpu]=$idx
    unset free_cpus[$last_idx]
    unset idx_map[$cpu]
done < <(
    ps -eo pid= 2>/dev/null | while read -r pid; do
        [[ $pid =~ ^[0-9]+$ ]] || continue
        [[ -n ${keep_pids[$pid]} ]] && continue
        mask=$(taskset -pc "$pid" 2>/dev/null | awk -F':' '/current affinity list/ {print $2}' || true)
        [[ -z $mask || "$mask" == " 0-319" || "$mask" == " 0-179" ]] && continue   # 未绑核或全核，空格不能删
        echo "$mask" | tr ',' '\n' | while read -r range; do
            if [[ $range =~ ^([0-9]+)-([0-9]+)$ ]]; then
                for ((c=${BASH_REMATCH[1]}; c<=${BASH_REMATCH[2]}; c++)); do
                    [[ $c -ge 0 && $c -le $((total_cpu-1)) ]] && echo "$c"
                done
            else
                [[ $range -ge 0 && $range -le $((total_cpu-1)) ]] && echo "$range"
            fi
        done
    done | sort -nu
)

remove_cpu() {
    local cpu=$1 idx=${idx_map[$cpu]}
    [[ -n $idx ]] || return 1
    last_idx=$((${#free_cpus[@]}-1))
    last_cpu=${free_cpus[$last_idx]}
    free_cpus[$idx]=$last_cpu
    idx_map[$last_cpu]=$idx
    unset free_cpus[$last_idx]
    unset idx_map[$cpu]
}

parse() {
    awk -F'[[:space:]]+' '{
        pid=tag=lr=""
        for(i=1;i<=NF;i++){
            if($i ~ /^pid=/)        pid=substr($i,5)
            if($i ~ /^tag=/)        tag=substr($i,5)
            if($i ~ /^local_rank=/) lr=substr($i,12)
        }
        print pid "|" tag "|" lr
    }' "$MARKER"
}

declare -A processed          

while IFS='|' read -r pid tag lr; do
    [[ $tag == "Worker" && -n $lr ]] || continue
    [[ -z ${processed[$pid]} ]] || continue      
    processed[$pid]=1
    # local_rank = 0 -> cpu 0-19, ...
    start=$(( lr*20 )); end=$(( start + 19 ))
    [[ $end -ge $((total_cpu-1)) ]] && end=$((total_cpu-1))
    cpu=""
    for ((c=end; c>=start; c--)); do
        [[ -n ${idx_map[$c]} ]] && { cpu=$c; break; }
    done
    if [[ -z $cpu ]]; then
        echo "[SKIP] No free cpu for Worker pid=$pid lr=$lr" | tee -a "$LOG_FILE"
        continue
    fi
    remove_cpu "$cpu"
    echo "Worker pid=$pid local_rank=$lr -> cpu $cpu" | tee -a "$LOG_FILE"
    taskset -pc "$cpu" "$pid" >/dev/null
done < <(parse | sort -t'|' -k3,3nr)

while IFS='|' read -r pid tag lr; do
    [[ $tag != "Worker" ]] || continue
    [[ "$ROLE" == "D" && $tag == "Tokenizer" ]] && continue
    [[ -z ${processed[$pid]} ]] || continue
    processed[$pid]=1
    if [[ ${#free_cpus[@]} -eq 0 ]]; then
        echo "[SKIP] No free cpu for $tag pid=$pid" | tee -a "$LOG_FILE"
        continue
    fi
    cpu=${free_cpus[0]}
    remove_cpu "$cpu"
    echo "$tag pid=$pid -> cpu $cpu" | tee -a "$LOG_FILE"
    taskset -pc "$cpu" "$pid" >/dev/null
done < <(parse)

echo "All done (ROLE=$ROLE). See $LOG_FILE"