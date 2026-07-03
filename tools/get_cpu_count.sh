#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# This script outputs the number of CPUs available in the current environment.
# It correctly handles container environments where nproc may return host CPU count.

# Function to get CPU count from cgroups v2
get_cpu_count_cgroup_v2() {
    local cpu_max_path="/sys/fs/cgroup/cpu.max"
    if [ -f "$cpu_max_path" ]; then
        local content quota period
        content=$(cat "$cpu_max_path")
        quota=$(echo "$content" | awk '{print $1}')
        period=$(echo "$content" | awk '{print $2}')
        if [ "$quota" = "max" ]; then
            return 1  # Fallback to system CPU count
        fi
        echo $(( quota / period ))
        return 0
    fi
    return 1
}

# Function to get CPU count from cgroups v1
get_cpu_count_cgroup_v1() {
    local quota_path="/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    local period_path="/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if [ -f "$quota_path" ] && [ -f "$period_path" ]; then
        local quota period
        quota=$(cat "$quota_path")
        period=$(cat "$period_path")
        if [ "$quota" -le 0 ] 2>/dev/null; then
            return 1  # quota <= 0 means unlimited, fallback
        fi
        echo $(( quota / period ))
        return 0
    fi
    return 1
}

# Function to get CPU count from CPU affinity
get_cpu_count_affinity() {
    if command -v nproc &> /dev/null; then
        # Try sched_getaffinity-based approach using python
        if command -v python3 &> /dev/null; then
            local count
            count=$(python3 -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null)
            if [ -n "$count" ] && [ "$count" -gt 0 ]; then
                echo "$count"
                return 0
            fi
        fi
    fi
    return 1
}

# Function to get CPU count from /proc/self/status
get_cpu_count_proc_status() {
    local status_path="/proc/self/status"
    if [ -f "$status_path" ]; then
        local cpus_allowed
        cpus_allowed=$(grep "Cpus_allowed:" "$status_path" 2>/dev/null | awk '{print $2}')
        if [ -n "$cpus_allowed" ]; then
            # Convert hex mask to binary and count 1s
            local hex_mask count
            hex_mask=$(echo "$cpus_allowed" | tr -d ',')
            if command -v python3 &> /dev/null; then
                count=$(python3 -c "print(bin(int('$hex_mask', 16)).count('1'))" 2>/dev/null)
                if [ -n "$count" ] && [ "$count" -gt 0 ]; then
                    echo "$count"
                    return 0
                fi
            fi
        fi
    fi
    return 1
}

# Get system default CPU count
get_system_cpu_count() {
    if [ -f /proc/cpuinfo ]; then
        grep -c "^processor" /proc/cpuinfo
    elif command -v nproc &> /dev/null; then
        nproc
    elif command -v sysctl &> /dev/null; then
        sysctl -n hw.ncpu 2>/dev/null || echo 1
    else
        echo 1
    fi
}

# Main function to get available CPU count
get_cpu_count() {
    local cpu_count

    # Try cgroups v2 first (most reliable for containers)
    cpu_count=$(get_cpu_count_cgroup_v2)
    if [ $? -eq 0 ] && [ -n "$cpu_count" ] && [ "$cpu_count" -gt 0 ]; then
        echo "$cpu_count"
        return 0
    fi

    # Try cgroups v1
    cpu_count=$(get_cpu_count_cgroup_v1)
    if [ $? -eq 0 ] && [ -n "$cpu_count" ] && [ "$cpu_count" -gt 0 ]; then
        echo "$cpu_count"
        return 0
    fi

    # Try CPU affinity
    cpu_count=$(get_cpu_count_affinity)
    if [ $? -eq 0 ] && [ -n "$cpu_count" ] && [ "$cpu_count" -gt 0 ]; then
        echo "$cpu_count"
        return 0
    fi

    # Try /proc/self/status
    cpu_count=$(get_cpu_count_proc_status)
    if [ $? -eq 0 ] && [ -n "$cpu_count" ] && [ "$cpu_count" -gt 0 ]; then
        echo "$cpu_count"
        return 0
    fi

    # Fallback to system CPU count
    get_system_cpu_count
}

# Output the CPU count
get_cpu_count