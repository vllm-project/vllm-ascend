#include "kv_transfer_client.h"
#include "connection_pool.h"
#include "common.h"
#include "kv_mem_resolver.h"

#include <iostream>
#include <cstring>
#include <sys/uio.h>
#include <algorithm>
#include <cerrno>
#include <set>
#include <map>
#include <thread>
#include <future>
#include <chrono>
#include <unordered_map>
#include <cstdlib>
#include <sys/socket.h>
#include <sys/time.h>
#include <deque>
#include <condition_variable>
#include <memory>
#include <mutex>

KVTransferClient& KVTransferClient::getInstance() {
    static KVTransferClient instance;
    return instance;
}

KVTransferClient::~KVTransferClient() { shutdown(); }


inline bool recv_n(int fd, void* buf, size_t len, int* out_errno = nullptr, size_t* out_read = nullptr) {
    size_t total = 0;
    auto* p = static_cast<unsigned char*>(buf);

    while (total < len) {
        ssize_t n = ::recv(fd, p + total, len - total, 0);
        if (n > 0) {
            total += static_cast<size_t>(n);
            continue;
        }
        if (n == 0) {
            if (out_errno) { *out_errno = 0; }
            if (out_read) { *out_read = total; }
            return false; // EOF
        }
        if (errno == EINTR) { continue; }

        if (out_errno) { *out_errno = errno; }
        if (out_read) { *out_read = total; }
        return false;
    }

    if (out_errno) { *out_errno = 0; }
    if (out_read) { *out_read = total; }
    return true;
}

static void set_sock_timeouts(int sockfd) {
    const char* t_env = std::getenv("KVC_TIMEOUT_MS");
    int ms = t_env ? std::max(1, std::atoi(t_env)) : 5000;
    timeval tv{}; tv.tv_sec = ms / 1000; tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

// Drain exactly n bytes into a small buffer
static bool drain_bytes(int fd, size_t n) {
    static thread_local std::vector<unsigned char> tmp(1 << 15); // 32KB
    size_t left = n;
    while (left > 0) {
        size_t chunk = std::min(left, tmp.size());
        int err = 0; size_t r = 0;
        if (!recv_n(fd, tmp.data(), chunk, &err, &r)) {
            if (err == 0) {
                std::cerr << "[XFER] drain: peer closed early, drained " << (n - left + r) << "/" << n << "\n";
            } else {
                std::cerr << "[XFER] drain: recv error errno=" << err << " (" << std::strerror(err) << ")\n";
            }
            return false;
        }
        left -= chunk;
    }
    return true;
}

// RAII guard to ensure the connection is released back to pool.
struct ConnGuard {
    uint32_t node_id;
    int fd;
    bool* healthy; // points to the flag in the caller
    ~ConnGuard() {
        if (fd >= 0) {
            ConnectionPool::getInstance().releaseConnection(node_id, fd, healthy ? *healthy : false);
        }
    }
    ConnGuard(uint32_t nid, int f, bool* h) : node_id(nid), fd(f), healthy(h) {}
    ConnGuard(const ConnGuard&) = delete;
    ConnGuard& operator=(const ConnGuard&) = delete;
};

// One-block transfer on a dedicated connection; copy only [off, off+seg) into D memory and drain the rest.
static bool transfer_one_block_segment(uint32_t node_id,
                                       const std::string& request_id,
                                       uint64_t p_block_id,
                                       uint64_t d_block_id,
                                       size_t off,
                                       size_t seg) {
    bool healthy = true;
    int fd = ConnectionPool::getInstance().acquireConnection(node_id, /*timeout_ms*/ 10000);
    if (fd < 0) {
        std::cerr << "[XFER] Node " << node_id << ": acquireConnection timeout/fail\n";
        return false;
    }
    ConnGuard guard(node_id, fd, &healthy);

    set_sock_timeouts(fd);

    // Build request for a single P block
    std::vector<uint64_t> p_blocks{p_block_id};
    auto req_data = serialize_request(request_id, node_id, p_blocks);
    if (!send_all(fd, req_data.data(), req_data.size())) {
        int e = errno;
        std::cerr << "[XFER] Node " << node_id << ": send_all failed errno=" << e << " (" << std::strerror(e) << ")\n";
        healthy = false; return false;
    }

    // Read response header and body
    MessageHeader rh{};
    int err = 0; size_t nread = 0;
    if (!recv_n(fd, &rh, sizeof(rh), &err, &nread)) {
        if (err == 0) {
            std::cerr << "[XFER] Node " << node_id << ": EOF during header, read "
                      << nread << "/" << sizeof(rh) << "\n";
        } else {
            std::cerr << "[XFER] Node " << node_id << ": recv header error errno=" << err
                      << " (" << std::strerror(err) << ")\n";
        }
        healthy = false; return false;
    }
    if (rh.magic != PROTOCOL_MAGIC || rh.type != MessageType::RESPONSE) {
        std::cerr << "[XFER] Node " << node_id << ": invalid response header\n";
        healthy = false; return false;
    }
    std::vector<uint8_t> body(rh.body_length);
    err = 0; nread = 0;
    if (!recv_n(fd, body.data(), body.size(), &err, &nread)) {
        if (err == 0) {
            std::cerr << "[XFER] Node " << node_id << ": EOF during body, read "
                      << nread << "/" << body.size() << "\n";
        } else {
            std::cerr << "[XFER] Node " << node_id << ": recv body error errno=" << err
                      << " (" << std::strerror(err) << ")\n";
        }
        healthy = false; return false;
    }

    std::string resp_req; uint32_t resp_node = 0; ErrorCode ec = ErrorCode::INTERNAL_ERROR;
    std::vector<uint64_t> resp_p_blocks; TensorMetadata meta{};
    if (!parse_response(body, resp_req, resp_node, ec, resp_p_blocks, meta)) {
        std::cerr << "[XFER] Node " << node_id << ": parse_response failed\n";
        healthy = false; return false;
    }
    if (ec != ErrorCode::SUCCESS || resp_req != request_id) {
        std::cerr << "[XFER] Node " << node_id << ": error code=" << (uint32_t)ec
                  << " resp_req='" << resp_req << "'\n";
        healthy = false; return false;
    }
    if (resp_p_blocks.size() != 1 || resp_p_blocks[0] != p_block_id) {
        std::cerr << "[XFER] Node " << node_id << ": p_blocks mismatch\n";
        healthy = false; return false;
    }
    size_t full = static_cast<size_t>(meta.total_size);
    if (off + seg > full) {
        std::cerr << "[XFER] Node " << node_id << ": segment out of range: off=" << off << " seg=" << seg
                  << " full=" << full << "\n";
        healthy = false; return false;
    }

    // Locate D memory
    kvmem::BlockView dv{};
    if (!kvmem::d_get_block(d_block_id, dv) || !dv.ptr || dv.bytes == 0) {
        std::cerr << "[XFER] Node " << node_id << ": D block missing " << d_block_id << "\n";
        healthy = false; return false;
    }
    if (off + seg > dv.bytes) {
        std::cerr << "[XFER] Node " << node_id << ": D segment out of range dv.bytes=" << dv.bytes << "\n";
        healthy = false; return false;
    }
    unsigned char* dst = static_cast<unsigned char*>(dv.ptr) + off;

    // Drain prefix, copy segment, drain suffix
    if (off > 0) {
        if (!drain_bytes(fd, off)) { healthy = false; return false; }
    }
    if (seg > 0) {
        int rc_err = 0; size_t rc_read = 0;
        if (!recv_n(fd, dst, seg, &rc_err, &rc_read)) {
            if (rc_err == 0) {
                std::cerr << "[XFER] Node " << node_id << ": EOF during seg, read " << rc_read << "/" << seg << "\n";
            } else {
                std::cerr << "[XFER] Node " << node_id << ": recv seg error errno=" << rc_err
                          << " (" << std::strerror(rc_err) << ")\n";
            }
            healthy = false; return false;
        }
    }
    size_t suffix = full - off - seg;
    if (suffix > 0) {
        if (!drain_bytes(fd, suffix)) { healthy = false; return false; }
    }
    return true;
}

// ========== Request ==========
bool KVTransferClient::requestKVTransfer(const KVRequest& request) {
    std::cout << "Starting KV transfer for request: " << request.request_id << std::endl << std::flush;

    // Collect P nodes involved and sort for stable node_index
    std::vector<uint32_t> p_node_ids = request.getAllPNodeIds();
    std::sort(p_node_ids.begin(), p_node_ids.end());
    const size_t total_nodes = p_node_ids.size();
    std::cout << "[XFER] Involved nodes: " << total_nodes << std::endl << std::flush;
    if (total_nodes == 0) {
        if (request.callback) request.callback(false, "No P nodes");
        return false;
    }

    // Build block-segment tasks (one task per mapping)
    struct SegTask {
        uint32_t node_id;
        uint64_t p_block_id;
        uint64_t d_block_id;
        size_t off;
        size_t seg;
    };
    std::vector<SegTask> tasks;
    tasks.reserve(request.block_mappings.size());

    // Precompute node_index map
    std::unordered_map<uint32_t, size_t> node_index;
    for (size_t i = 0; i < p_node_ids.size(); ++i) node_index[p_node_ids[i]] = i;

    // For each mapping, compute segment for that node
    for (const auto& m : request.block_mappings) {
        auto it = node_index.find(m.p_node_id);
        if (it == node_index.end()) {
            std::cerr << "[XFER] mapping references unknown node_id " << m.p_node_id << "\n";
            if (request.callback) request.callback(false, "Unknown node in mapping");
            return false;
        }
        size_t idx = it->second;
        kvmem::BlockView dv{};
        if (!kvmem::d_get_block(m.d_block_id, dv) || !dv.ptr || dv.bytes == 0) {
            std::cerr << "[XFER] No D memory for D block " << m.d_block_id << "\n";
            if (request.callback) request.callback(false, "D memory missing");
            return false;
        }
        if (dv.bytes % total_nodes != 0) {
            std::cerr << "[XFER] D block bytes " << dv.bytes << " not divisible by total_nodes " << total_nodes << "\n";
            if (request.callback) request.callback(false, "D block size mismatch");
            return false;
        }
        size_t seg = dv.bytes / total_nodes;
        size_t off = idx * seg;
        tasks.push_back(SegTask{ m.p_node_id, m.p_block_id, m.d_block_id, off, seg });
    }

    // Launch tasks in parallel. Concurrency is effectively limited by per-node connection pools.
    std::vector<std::future<bool>> futs;
    futs.reserve(tasks.size());
    for (const auto& t : tasks) {
        futs.emplace_back(std::async(std::launch::async, [rid = request.request_id, t]() {
            return transfer_one_block_segment(t.node_id, rid, t.p_block_id, t.d_block_id, t.off, t.seg);
        }));
    }

    bool all_success = true;
    int ok_cnt = 0;
    for (auto& f : futs) {
        bool ok = false;
        try { ok = f.get(); }
        catch (const std::exception& e) { std::cerr << "[XFER] task exception: " << e.what() << "\n"; ok = false; }
        catch (...) { std::cerr << "[XFER] task unknown exception\n"; ok = false; }
        if (!ok) { all_success = false; }
        else ok_cnt++;
    }

    std::cout << "[XFER] Summary: " << ok_cnt << "/" << tasks.size() << " block-segments successful\n" << std::flush;
    if (request.callback) request.callback(all_success, all_success ? "Success" : "Some segments failed");
    return all_success;
}

void KVTransferClient::shutdown() {
    std::cout << "Shutting down KV transfer client..." << std::endl << std::flush;
    ConnectionPool::getInstance().closeAll();
    std::cout << "KV transfer client shutdown completed" << std::endl << std::flush;
}

bool KVTransferClient::startWorkers() { return true; }
void KVTransferClient::stopWorkers() {}
bool KVTransferClient::workersEnabled() const { return true; }