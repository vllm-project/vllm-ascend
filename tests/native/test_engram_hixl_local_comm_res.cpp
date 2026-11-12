#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <unistd.h>

#include "acl/acl.h"
#include "hixl/hixl.h"
#include "../../csrc/engram_hixl/native.cpp"

namespace {
size_t largest_allocation = 0;

std::string write_file(const std::string& path, size_t bytes) {
  std::ofstream output(path, std::ios::binary);
  assert(output);
  if (bytes) {
    const std::string chunk(4096, 'x');
    size_t written = 0;
    while (written < bytes && written < (1u << 20) + 1) {
      const auto count = std::min(chunk.size(), bytes - written);
      output.write(chunk.data(), static_cast<std::streamsize>(count));
      written += count;
    }
    output.seekp(static_cast<std::streamoff>(bytes - 1));
    if (written < bytes) output.put('x');
  }
  output.close();
  assert(output);
  return path;
}

void clear_env() {
  unsetenv("ENGRAM_HIXL_LOCAL_COMM_RES");
  unsetenv("VLLM_ASCEND_ENGRAM_HIXL_LISTEN_PORT");
}

void reset_stubs() {
  engram_hixl_cpu_test::acl_calls = 0;
  hixl::cpu_test::initialize_calls = 0;
  hixl::cpu_test::local_comm_res.clear();
}

void expect_invalid_resource(const std::string& path, const char* expected_error) {
  clear_env();
  reset_stubs();
  setenv("ENGRAM_HIXL_LOCAL_COMM_RES", path.c_str(), 1);
  void* engine = reinterpret_cast<void*>(1);
  largest_allocation = 0;
  assert(engram_hixl_create("cpu-test", 0, &engine) == -1);
  assert(engine == nullptr);
  assert(std::string(engram_hixl_error()) == expected_error);
  assert(hixl::cpu_test::initialize_calls == 0);
}
}  // namespace

void* operator new(std::size_t size) {
  largest_allocation = size > largest_allocation ? size : largest_allocation;
  if (void* memory = std::malloc(size)) return memory;
  throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

int main() {
  const std::string prefix = "/tmp/engram-hixl-local-comm-res-" + std::to_string(getpid());
  const auto empty = write_file(prefix + "-empty", 0);
  const auto one_byte = write_file(prefix + "-one", 1);
  const auto limit = write_file(prefix + "-limit", 1u << 20);
  const auto oversized = write_file(prefix + "-large", 32u << 20);

  expect_invalid_resource(empty, "ENGRAM_HIXL_LOCAL_COMM_RES must be 1 byte to 1 MiB");
  expect_invalid_resource(prefix + "-missing", "cannot read ENGRAM_HIXL_LOCAL_COMM_RES");
  expect_invalid_resource(oversized, "ENGRAM_HIXL_LOCAL_COMM_RES must be 1 byte to 1 MiB");
  assert(largest_allocation < (2u << 20));

  clear_env();
  reset_stubs();
  setenv("ENGRAM_HIXL_LOCAL_COMM_RES", one_byte.c_str(), 1);
  void* engine = nullptr;
  assert(engram_hixl_create("cpu-test", 0, &engine) == 0);
  assert(engine != nullptr);
  assert(hixl::cpu_test::initialize_calls == 1);
  assert(hixl::cpu_test::local_comm_res == "x");
  assert(engram_hixl_close(engine) == 0);

  clear_env();
  reset_stubs();
  setenv("ENGRAM_HIXL_LOCAL_COMM_RES", limit.c_str(), 1);
  engine = nullptr;
  assert(engram_hixl_create("cpu-test", 0, &engine) == 0);
  assert(engine != nullptr);
  assert(hixl::cpu_test::initialize_calls == 1);
  assert(hixl::cpu_test::local_comm_res.size() == (1u << 20));
  assert(engram_hixl_close(engine) == 0);

  clear_env();
  reset_stubs();
  setenv("ENGRAM_HIXL_LOCAL_COMM_RES", one_byte.c_str(), 1);
  setenv("VLLM_ASCEND_ENGRAM_HIXL_LISTEN_PORT", "12345", 1);
  engine = reinterpret_cast<void*>(1);
  assert(engram_hixl_create("cpu-test", 0, &engine) == -1);
  assert(engine == nullptr);
  assert(std::string(engram_hixl_error()) ==
         "VLLM_ASCEND_ENGRAM_HIXL_LISTEN_PORT and ENGRAM_HIXL_LOCAL_COMM_RES are mutually exclusive");
  assert(hixl::cpu_test::initialize_calls == 0);

  std::remove(empty.c_str());
  std::remove(one_byte.c_str());
  std::remove(limit.c_str());
  std::remove(oversized.c_str());
  clear_env();
  std::puts("production native.cpp CPU harness passed");
}
