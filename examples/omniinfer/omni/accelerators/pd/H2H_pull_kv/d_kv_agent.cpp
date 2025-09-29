#include "connection_pool.h"
#include "kv_transfer_client.h"
#include "kv_mem_resolver.h"
#include "common.h"

#include <zmq.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <optional>
#include <csignal>
#include <mutex>
#include <cstdint>
#include <chrono>
#include <cctype>

// ----------------- Logger & Helper -----------------
static void usage() {
    std::cout <<
    "d_kv_agent - D-side KV pull agent with ZeroMQ ROUTER server\n"
    "Required:\n"
    "  --nodes ip:port:node_id;ip:port:node_id;...\n"
    "  --d_mmap <path>\n"
    "  --dtype <bf16|f16|f32|f64|i8|i16|i32|i64>\n"
    "  --offset_dtype <elements_per_d_block>\n"
    "  --start_offset <0>\n"
    "  --zmq_bind tcp://127.0.0.1:5555\n"
    "Optional:\n"
    "  --conns_per_node <N>    (default 64; overrides env KVC_CONNS_PER_NODE)\n"
    "Env (fallbacks if not specified by flags):\n"
    "  KVC_CONNS_PER_NODE: number of TCP connections per P node\n"
    "  D_AGENT_WORKERS:    number of pull_kv worker threads (default: 4)\n"
    << std::endl;
}

// ----------------- Routing envelope: supports multiple identities plus an optional empty delimiter frame. -----------------
struct RouteEnvelope {
    std::vector<std::vector<uint8_t>> identities; // 0..N-1: ROUTER identity stack
    bool has_delimiter{false};                    // if add empty delimiter after identities
};

// ----------------- ZMQ send/rec helper -----------------
static bool recv_multipart_all(void* sock, std::vector<std::vector<uint8_t>>& frames) {
    frames.clear();
    while (true) {
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        int rc = zmq_msg_recv(&msg, sock, 0);
        if (rc < 0) { zmq_msg_close(&msg); return false; }
        std::vector<uint8_t> buf(
            static_cast<uint8_t*>(zmq_msg_data(&msg)),
            static_cast<uint8_t*>(zmq_msg_data(&msg)) + zmq_msg_size(&msg)
        );
        frames.emplace_back(std::move(buf));
        int more = 0; size_t more_size = sizeof(more);
        zmq_getsockopt(sock, ZMQ_RCVMORE, &more, &more_size);
        zmq_msg_close(&msg);
        if (!more) break;
    }
    return true;
}

// ROUTER: receives the full message → splits it into [routing envelope][payload...]
static bool router_recv_envelope_and_payload(void* router, RouteEnvelope& out_route,
                                             std::vector<std::vector<uint8_t>>& payload_frames) {
    out_route.identities.clear();
    out_route.has_delimiter = false;
    payload_frames.clear();

    std::vector<std::vector<uint8_t>> frames;
    if (!recv_multipart_all(router, frames)) return false;
    if (frames.empty()) return false;

    // search for the first empty delimiter
    size_t delim_idx = frames.size();
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].empty()) { delim_idx = i; break; }
    }

    if (delim_idx < frames.size()) {
        // has empty delimiter：0..delim-1 are for identities，delim is for empty frame，delim+1.. is for payload
        out_route.has_delimiter = true;
        out_route.identities.assign(frames.begin(), frames.begin() + delim_idx);
        payload_frames.assign(frames.begin() + delim_idx + 1, frames.end());
    } else {
        // no empty delimiter：frames[0] is single identity，1.. are payload（REQ->ROUTER）
        out_route.has_delimiter = false;
        out_route.identities.clear();
        out_route.identities.push_back(frames.front());
        if (frames.size() > 1) {
            payload_frames.assign(frames.begin() + 1, frames.end());
        }
    }
    return true;
}

// ROUTER: sending [routing envelope][payload...]
static bool router_send_with_envelope(void* router, const RouteEnvelope& route,
                                      const std::vector<std::pair<const void*, size_t>>& payload_frames) {
    // replay identities
    for (const auto& id : route.identities) {
        if (zmq_send(router, id.data(), id.size(), ZMQ_SNDMORE) < 0) return false;
    }
    // optional empty delimiter frame
    if (route.has_delimiter) {
        if (zmq_send(router, nullptr, 0, ZMQ_SNDMORE) < 0) return false;
    }
    // sending payload
    for (size_t i = 0; i < payload_frames.size(); ++i) {
        int flags = (i + 1 < payload_frames.size()) ? ZMQ_SNDMORE : 0;
        if (zmq_send(router, payload_frames[i].first, payload_frames[i].second, flags) < 0) return false;
    }
    return true;
}

// ----------------- Common parse helpers -----------------
static inline std::string frame_to_string(const std::vector<uint8_t>& v) {
    return std::string(reinterpret_cast<const char*>(v.data()), v.size());
}

static kvmem::ScalarType parse_dtype_cpp(const std::string& s) {
    std::string t = s;
    for (auto& c: t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "bf16") return kvmem::ScalarType::BF16;
    if (t == "f16" ) return kvmem::ScalarType::F16;
    if (t == "f32" ) return kvmem::ScalarType::F32;
    if (t == "f64" ) return kvmem::ScalarType::F64;
    if (t == "i8"  ) return kvmem::ScalarType::I8;
    if (t == "i16" ) return kvmem::ScalarType::I16;
    if (t == "i32" ) return kvmem::ScalarType::I32;
    if (t == "i64" ) return kvmem::ScalarType::I64;
    return kvmem::ScalarType::BF16;
}

static bool split_nodes(const std::string& nodes_str, std::vector<std::string>& out_specs) {
    out_specs.clear();
    std::stringstream ss(nodes_str);
    std::string item;
    while (std::getline(ss, item, ';')) {
        if (item.empty()) continue;
        size_t p1 = item.find(':'), p2 = (p1==std::string::npos)?std::string::npos:item.find(':', p1+1);
        if (p1 == std::string::npos || p2 == std::string::npos) {
            std::cerr << "[D_AGENT] invalid node spec: " << item << std::endl; return false;
        }
        out_specs.push_back(item);
    }
    return !out_specs.empty();
}

static inline uint64_t load_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

static bool parse_u64_array(const std::vector<uint8_t>& bytes, std::vector<uint64_t>& out) {
    if (bytes.size() % 8 != 0) return false;
    size_t n = bytes.size() / 8;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = load_u64_le(&bytes[i*8]);
    return true;
}

// ----------------- Concurrent queue -----------------
template <typename T>
class ConcurrentQueue {
public:
    void push(T&& v) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace_back(std::move(v));
        }
        cv_.notify_one();
    }
    bool wait_pop(T& out, std::atomic<bool>& stop_flag) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return stop_flag.load() || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    void notify_all() { cv_.notify_all(); }
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> q_;
};

// ----------------- task and result -----------------
struct PullTask {
    RouteEnvelope route;     // Full routing envelope needed for the reply
    std::string request_id;
    uint32_t cluster_id{0};
    uint32_t cluster_size{0};
    std::vector<uint64_t> p_ids;
    std::vector<uint64_t> d_ids;
};

struct PullResult {
    RouteEnvelope route;     // Full routing envelope needed for the reply
    std::string request_id;
    bool ok{false};
    std::string msg; // "Success" or error detail
};

// ----------------- Global stop flag -----------------
static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop.store(true); }

static bool perform_cluster_pull(const std::string& request_id,
                                 uint32_t cluster_id,
                                 uint32_t cluster_size,
                                 const std::vector<uint64_t>& p_block_ids,
                                 const std::vector<uint64_t>& d_block_ids) {
    if (p_block_ids.size() != d_block_ids.size()) {
        std::cerr << "[D_AGENT] size mismatch p=" << p_block_ids.size()
                  << " d=" << d_block_ids.size() << std::endl;
        return false;
    }

    auto node_ids_all = ConnectionPool::getInstance().getAllNodeIds();
    if (node_ids_all.empty()) {
        std::cerr << "[D_AGENT] No P nodes in pool\n";
        return false;
    }
    std::sort(node_ids_all.begin(), node_ids_all.end());

    uint32_t begin_id = cluster_id * cluster_size;
    uint32_t end_id   = begin_id + cluster_size;
    std::vector<uint32_t> cluster_nodes;
    for (uint32_t nid : node_ids_all) if (nid >= begin_id && nid < end_id) cluster_nodes.push_back(nid);
    if (cluster_nodes.size() != cluster_size) {
        std::cerr << "[D_AGENT] cluster " << cluster_id << " want " << cluster_size
                  << " got " << cluster_nodes.size() << std::endl;
        return false;
    }

    // construct the request
    KVRequest req{request_id};
    const size_t A = d_block_ids.size();
    req.d_blocks.reserve(A);
    for (size_t k = 0; k < A; ++k) req.d_blocks.emplace_back(d_block_ids[k], nullptr, 0);
    for (size_t k = 0; k < A; ++k) {
        for (uint32_t nid : cluster_nodes) {
            req.block_mappings.emplace_back(nid, p_block_ids[k], d_block_ids[k]);
        }
    }

    // Ensure the connection to the target node is ready (multi-conn pool already exists)
    for (uint32_t nid : cluster_nodes) {
        if (!ConnectionPool::getInstance().isConnected(nid)) {
            std::cout << "[D_AGENT] Reconnecting node " << nid << std::endl;
            if (!ConnectionPool::getInstance().reconnect(nid)) {
                std::cerr << "[D_AGENT] reconnect failed for node " << nid << std::endl;
                return false;
            }
        }
    }

    return KVTransferClient::getInstance().requestKVTransfer(req);
}

// ----------------- main program -----------------
int main(int argc, char** argv) {
    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);

    std::string nodes_str, d_mmap, dtype_str = "bf16", zmq_endpoint = "tcp://127.0.0.1:5555";
    size_t offset_dtype = 0, start_offset = 0;
    uint64_t direct_id_base = 0;
    int conns_per_node_arg = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nodes" && i+1 < argc) nodes_str = argv[++i];
        else if (a == "--d_mmap" && i+1 < argc) d_mmap = argv[++i];
        else if (a == "--dtype" && i+1 < argc) dtype_str = argv[++i];
        else if (a == "--offset_dtype" && i+1 < argc) offset_dtype = static_cast<size_t>(std::stoull(argv[++i]));
        else if (a == "--start_offset" && i+1 < argc) start_offset = static_cast<size_t>(std::stoull(argv[++i]));
        else if (a == "--direct_id_base" && i+1 < argc) direct_id_base = std::stoull(argv[++i]);
        else if (a == "--zmq_bind" && i+1 < argc) zmq_endpoint = argv[++i];
        else if (a == "--conns_per_node" && i+1 < argc) conns_per_node_arg = std::max(1, std::stoi(argv[++i]));
        else if (a == "--help") { usage(); return 0; }
    }

    if (nodes_str.empty() || d_mmap.empty() || offset_dtype == 0) {
        usage();
        return 2;
    }

    std::vector<std::string> node_specs;
    if (!split_nodes(nodes_str, node_specs)) {
        std::cerr << "[D_AGENT] invalid --nodes" << std::endl;
        return 2;
    }

    auto dtype = parse_dtype_cpp(dtype_str);
    if (!kvmem::init_d_side_from_mmap(d_mmap, dtype, offset_dtype, start_offset)) {
        std::cerr << "[D_AGENT] init_d_side_from_mmap failed" << std::endl;
        return 2;
    }
    if (!kvmem::d_set_id_mode(/*direct*/1, direct_id_base)) {
        std::cerr << "[D_AGENT] d_set_id_mode(direct) failed" << std::endl;
        return 2;
    }
    std::cout << "[D_AGENT] D mmap OK path=" << d_mmap
              << " dtype=" << dtype_str
              << " offset_dtype=" << offset_dtype
              << " start_offset=" << start_offset
              << " direct_id_base=" << direct_id_base << std::endl;

    ConnectionPool::ConnectOptions copts = ConnectionPool::ConnectOptions::fromEnv();
    copts.allow_partial = true;
    copts.overall_wait_ms = 20000;
    if (conns_per_node_arg > 0) {
        copts.conns_per_node = conns_per_node_arg; // override env with CLI
    }

    bool init_ok = ConnectionPool::getInstance().initialize(node_specs, copts);
    if (!init_ok) {
        std::cerr << "[D_AGENT] ConnectionPool initialize failed";
        if (!copts.allow_partial) {
            std::cerr << " (partial not allowed), exiting\n";
            return 2;
        }
        std::cerr << " (allow_partial), continue with subset\n";
    }

    KVTransferClient::getInstance().startWorkers();

    int workers = 4;
    if (const char* env = std::getenv("D_AGENT_WORKERS")) {
        try { workers = std::max(1, std::stoi(env)); } catch (...) {}
    }

    ConcurrentQueue<PullTask> task_q;
    ConcurrentQueue<PullResult> result_q;

    // ZeroMQ context and ROUTER socket
    void* ctx = zmq_ctx_new();
    void* router = zmq_socket(ctx, ZMQ_ROUTER);
    int hwm = 1000;
    zmq_setsockopt(router, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(router, ZMQ_RCVHWM, &hwm, sizeof(hwm));
    int sndtimeo = 0; // Non-blocking send: returns EAGAIN immediately on back-pressure
    zmq_setsockopt(router, ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
    if (::zmq_bind(router, zmq_endpoint.c_str()) != 0) {
        std::cerr << "[D_AGENT] zmq_bind failed: " << zmq_endpoint
                  << " err=" << zmq_strerror(zmq_errno()) << std::endl;
        zmq_close(router); zmq_ctx_term(ctx);
        return 2;
    }
    std::cout << "[D_AGENT] READY (ROUTER) bind=" << zmq_endpoint
              << " pull_kv_workers=" << workers
              << " conns_per_node=" << copts.conns_per_node
              << std::endl;

    // Worker thread: processes tasks only, pushes results into result_q, never touches router directly
    std::vector<std::thread> worker_threads;
    std::atomic<bool> stop_workers{false};
    for (int i = 0; i < workers; ++i) {
        worker_threads.emplace_back([&](){
            while (!stop_workers.load()) {
                PullTask task;
                if (!task_q.wait_pop(task, stop_workers)) {
                    if (stop_workers.load()) break;
                    continue;
                }
                PullResult res;
                res.route = std::move(task.route);
                res.request_id = task.request_id;
                try {
                    bool ok = perform_cluster_pull(task.request_id, task.cluster_id, task.cluster_size,
                                                   task.p_ids, task.d_ids);
                    res.ok = ok;
                    res.msg = ok ? "Success" : "pull failed";
                } catch (const std::exception& e) {
                    res.ok = false;
                    res.msg = std::string("exception: ") + e.what();
                } catch (...) {
                    res.ok = false;
                    res.msg = "unknown exception";
                }
                result_q.push(std::move(res));
            }
        });
    }

    // Single I/O thread: exclusively owns the router, handles receiving requests and sending back results
    // (routing envelope echoed unchanged).
    std::thread io_thread([&](){
        std::deque<PullResult> pending;

        while (!g_stop.load()) {
            // 1) poll new request
            zmq_pollitem_t items[] = { { router, 0, ZMQ_POLLIN, 0 } };
            int rc = zmq_poll(items, 1, 50);
            if (rc < 0 && zmq_errno() != EINTR) {
                std::cerr << "[D_AGENT] zmq_poll(io) error: " << zmq_strerror(zmq_errno()) << std::endl;
            }

            if (rc > 0 && (items[0].revents & ZMQ_POLLIN)) {
                RouteEnvelope route;
                std::vector<std::vector<uint8_t>> frames;
                if (!router_recv_envelope_and_payload(router, route, frames)) {
                    std::cerr << "[D_AGENT] router recv failed\n";
                } else {
                    if (frames.size() != 6) {
                        const char* err = "ERR"; const char* msg = "invalid message format";
                        std::string maybe_req = frames.size() > 1 ? frame_to_string(frames[1]) : "";
                        if (!maybe_req.empty()) {
                            if (!router_send_with_envelope(router, route, {
                                    {err, strlen(err)},
                                    {maybe_req.data(), maybe_req.size()},
                                    {msg, strlen(msg)}
                                })) {
                                std::cerr << "[D_AGENT] router send(err invalid format) failed: "
                                          << zmq_strerror(zmq_errno()) << std::endl;
                            }
                        } else {
                            if (!router_send_with_envelope(router, route, {
                                    {err, strlen(err)},
                                    {msg, strlen(msg)}
                                })) {
                                std::cerr << "[D_AGENT] router send(err no req_id) failed: "
                                          << zmq_strerror(zmq_errno()) << std::endl;
                            }
                        }
                    } else {
                        std::string op = frame_to_string(frames[0]);
                        std::string req_id = frame_to_string(frames[1]);
                        if (op != "pull_kv") {
                            const char* err = "ERR"; const char* msg = "unknown op";
                            if (!router_send_with_envelope(router, route, {
                                    {err, strlen(err)},
                                    {req_id.data(), req_id.size()},
                                    {msg, strlen(msg)}
                                })) {
                                std::cerr << "[D_AGENT] router send(err unknown op) failed: "
                                          << zmq_strerror(zmq_errno()) << std::endl;
                            }
                        } else {
                            std::cout << "[D_AGENT] received request, op=" << op
                                      << " request_id=" << req_id << std::endl;

                            PullTask task;
                            task.route = std::move(route);
                            task.request_id = std::move(req_id);
                            task.cluster_id = static_cast<uint32_t>(std::strtoul(frame_to_string(frames[2]).c_str(), nullptr, 10));
                            task.cluster_size = static_cast<uint32_t>(std::strtoul(frame_to_string(frames[3]).c_str(), nullptr, 10));
                            if (!parse_u64_array(frames[4], task.p_ids) || !parse_u64_array(frames[5], task.d_ids)) {
                                const char* err = "ERR";
                                const char* msg = "bad id arrays";
                                if (!router_send_with_envelope(router, task.route, {
                                        {err, strlen(err)},
                                        {task.request_id.data(), task.request_id.size()},
                                        {msg, strlen(msg)}
                                    })) {
                                    std::cerr << "[D_AGENT] router send(err bad id arrays) failed: "
                                              << zmq_strerror(zmq_errno()) << std::endl;
                                }
                            } else {
                                task_q.push(std::move(task));
                            }
                        }
                    }
                }
            }

            // 2) send reply
            for (;;) {
                PullResult res;
                if (!result_q.try_pop(res)) break;
                pending.emplace_back(std::move(res));
            }
            size_t pend_sz = pending.size();
            for (size_t i = 0; i < pend_sz; ++i) {
                PullResult res = std::move(pending.front());
                pending.pop_front();
                const char* status = res.ok ? "OK" : "ERR";
                std::cout << "[D_AGENT] sending reply to request_id=" << res.request_id << std::endl;
                if (!router_send_with_envelope(router, res.route, {
                        {status, std::strlen(status)},
                        {res.request_id.data(), res.request_id.size()},
                        {res.msg.data(), res.msg.size()}
                    })) {
                    int err = zmq_errno();
                    if (err == EAGAIN) {
                        pending.emplace_back(std::move(res));
                        break;
                    } else {
                        std::cerr << "[D_AGENT] router send failed for request_id=" << res.request_id
                                  << " err=" << zmq_strerror(err) << std::endl;
                    }
                }
            }
        }
    });

    // Main thread waits for the stop signal.
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Cleanup / shutdown
    stop_workers.store(true);
    task_q.notify_all();

    if (io_thread.joinable()) io_thread.join();
    for (auto& th : worker_threads) {
        if (th.joinable()) th.join();
    }

    zmq_close(router);
    zmq_ctx_term(ctx);

    KVTransferClient::getInstance().stopWorkers();
    ConnectionPool::getInstance().closeAll();
    kvmem::shutdown_all();
    return 0;
}