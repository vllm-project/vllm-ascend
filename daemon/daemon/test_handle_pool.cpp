#include "inc/handle_pool.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using mdaemon::HandlePool;

int main() {
    std::array<uint64_t, 3> page_sizes = {4ULL * 1024 * 1024, 8ULL * 1024 * 1024, 2ULL * 1024 * 1024};
    std::array<int32_t, 4> devices = {0, 1, 2, 3};

    for (uint64_t page_size : page_sizes) {
        HandlePool pool(page_size);
        uint64_t total_bytes = page_size * 8; // keep a few handles available per device

        std::cout << "Testing page size " << page_size / (1024 * 1024) << " MB" << std::endl;
        for (int32_t device_id : devices) {
            try {
                pool.initializeDevice(device_id, total_bytes);
                std::cout << "  Initialized device " << device_id << std::endl;
            } catch (const std::exception& ex) {
                std::cerr << "Failed to initialize device " << device_id << ": " << ex.what() << std::endl;
                return 1;
            }

            size_t available_before = pool.available(device_id);
            std::cout << "    Available handles: " << available_before << std::endl;
            if (available_before == 0) {
                std::cerr << "No handles available after initialization for device " << device_id << std::endl;
                return 1;
            }

            auto handle = pool.acquire(device_id);
            if (!handle.valid()) {
                std::cerr << "Failed to acquire a valid handle for device " << device_id << std::endl;
                return 1;
            }
            else {
                std::cout << "    Acquired handle with granularity " << handle.granularity() / (1024 * 1024) << " MB" << std::endl;
            }

            uint64_t shareable = handle.shareableHandle();
            if (shareable == std::numeric_limits<uint64_t>::max()) {
                std::cerr << "shareable handle was not updated for device " << device_id << std::endl;
                return 1;
            }

            pool.release(device_id, std::move(handle));
            std::cout << "    Released handle back to pool" << std::endl;

            size_t available_after = pool.available(device_id);
            if (available_after != available_before) {
                std::cerr << "Handle count mismatch for device " << device_id
                          << " (before=" << available_before << " after=" << available_after << ")"
                          << std::endl;
                return 1;
            }

            std::cout << "  device " << device_id << " passed" << std::endl;
        }

        // additional extend/remove verification on a subset of chips
        std::array<int32_t, 2> extend_devices = {0, 2};
        for (int32_t device_id : extend_devices) {
            size_t available_before = pool.available(device_id);
            size_t extend_count = 3;
            if (!pool.extendHandles(device_id, extend_count)) {
                std::cerr << "extendHandles failed for device " << device_id << std::endl;
                return 1;
            }
            size_t available_extended = pool.available(device_id);
            if (available_extended != available_before + extend_count) {
                std::cerr << "extendHandles did not add the expected handles for device " << device_id << std::endl;
                return 1;
            }

            size_t remove_count = 2;
            if (!pool.removeHandles(device_id, remove_count)) {
                std::cerr << "removeHandles failed for device " << device_id << std::endl;
                return 1;
            }
            size_t available_removed = pool.available(device_id);
            if (available_removed != available_extended - remove_count) {
                std::cerr << "removeHandles did not drop the expected handles for device " << device_id << std::endl;
                return 1;
            }

            std::cout << "  extend/remove pair succeeded for device " << device_id << std::endl;
        }

        // print the handle pool items
        for (int32_t device_id : devices) {
            size_t available = pool.available(device_id);
            std::cout << "  Final available handles for device " << device_id << ": " << available << std::endl;
            
            // handles print
            for (size_t i = 0; i < available; ++i) {
                auto handle = pool.acquire(device_id);
                std::cout << "    Handle " << i << ": granularity " << handle.granularity() / (1024 * 1024)
                          << " MB, shareable handle " << handle.shareableHandle() << std::endl;
                pool.release(device_id, std::move(handle));
            }
        }

        // shutdown pool
        pool.shutdown();
        std::cout << "  Shutdown pool for page size " << page_size / (1024 * 1024) << " MB" << std::endl;
    }

    std::cout << "test_handle_pool: all configurations succeed" << std::endl;
    return 0;
}