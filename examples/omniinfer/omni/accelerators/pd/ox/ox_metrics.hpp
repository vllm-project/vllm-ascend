// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

std::mutex g_output_mutex;
std::atomic<uint64_t> g_total_bytes_received{0};

std::chrono::steady_clock::time_point g_program_start_time;

inline void global_stats_update(size_t n)
{
    g_total_bytes_received += n;
}

awaitable<void> print_statistics()
{
    while (true)
    {
        asio::steady_timer timer(co_await asio::this_coro::executor);
        timer.expires_after(std::chrono::seconds(2));
        co_await timer.async_wait(use_awaitable);

        auto now = std::chrono::steady_clock::now();
        double global_bandwidth_mbps = (g_total_bytes_received) /
                                       (std::chrono::duration_cast<std::chrono::microseconds>(now - g_program_start_time).count() / 1e6) / 1e6;

        {
            std::lock_guard<std::mutex> lock(g_output_mutex);
            std::cout << "\nGlobal bandwidth: " << global_bandwidth_mbps
                      << " MB | Total data: " << g_total_bytes_received / (1024 * 1024) << " MB"
                      << std::flush;
        }
    }
}