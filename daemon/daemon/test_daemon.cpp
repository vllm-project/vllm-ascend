#include "inc/daemon.hpp"
#include "inc/handle_pool.hpp"
#include "inc/model_management.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

// define colors for better visibility in test output
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"

using namespace mdaemon;

namespace {

void test_handle_pool(Daemon& d, uint64_t granularity, int device_id) {
    constexpr int kCount = 10;
    HandlePool* pool = d.ensureHandlePool(granularity);
    d.initializeHandlePoolDevice(granularity, device_id, granularity * kCount); // allocate a bit extra
    std::vector<Handle> handles;
    handles.reserve(kCount);
    for (int i = 0; i < kCount; ++i) {
        Handle h = pool->acquire(device_id);
        handles.push_back(std::move(h));
    }
    std::cout << "[HandlePool] acquired " << kCount << " handles at granularity=" << granularity
              << " device=" << device_id << " available(after acquire)=" << pool->available(device_id) << "\n";
    // release back
    for (auto& h : handles) {
        pool->release(device_id, std::move(h));
    }
    std::cout << "[HandlePool] released " << kCount << " handles at granularity=" << granularity
              << " device=" << device_id << " available(after release)=" << pool->available(device_id) << "\n";

    const size_t available_before = pool->available(device_id);
    const size_t extend_count = 3;
    if (d.extendHandles(granularity, device_id, extend_count)) {
        const size_t available_extended = pool->available(device_id);
        std::cout << "[HandlePool] extended " << extend_count << " handles at granularity=" << granularity
                  << " device=" << device_id << " available(after extend)=" << available_extended << "\n";
    } else {
        std::cerr << "[HandlePool] extendHandles failed for granularity=" << granularity
                  << " device=" << device_id << "\n";
    }

    const size_t remove_count = 2;
    if (d.removeHandles(granularity, device_id, remove_count)) {
        const size_t available_removed = pool->available(device_id);
        std::cout << "[HandlePool] removed " << remove_count << " handles at granularity=" << granularity
                  << " device=" << device_id << " available(after remove)=" << available_removed << "\n";
    } else {
        std::cerr << "[HandlePool] removeHandles failed for granularity=" << granularity
                  << " device=" << device_id << "\n";
    }
}

} // namespace

int main() {
    std::cout << "[DaemonTest] starting daemon...\n";
    Daemon daemon;
    daemon.start();

    // HandlePool tests on device 0 and 1 for 2/4/8/16MB granularity
    const std::array<uint64_t, 4> granularities = {
        2ULL * 1024 * 1024,
        4ULL * 1024 * 1024,
        8ULL * 1024 * 1024,
        16ULL * 1024 * 1024
    };
    for (uint64_t granularity : granularities) {
        test_handle_pool(daemon, granularity, 0);
        test_handle_pool(daemon, granularity, 1);
    }
    daemon.printAll();

    std::atomic<bool> stop_requested{false};
    std::thread input_thread([&stop_requested]() {
        std::cout << COLOR_GREEN << "\n[DaemonTest] press 'q' then ENTER to stop." << COLOR_RESET << std::endl;
        for (char ch; std::cin >> ch;) {
            if (ch == 'q' || ch == 'Q') {
                stop_requested.store(true);
                break;
            }
        }
    });

    std::cout << "[DaemonTest] daemon loop running; launch model processes separately." << std::endl;
    while (!stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (input_thread.joinable()) {
        input_thread.join();
    }

    std::cout << "[DaemonTest] stopping daemon...\n";
    daemon.stop();

    std::cout << "[DaemonTest] remaining models: " << daemon.listModelIds().size() << "\n";
    return 0;
}
