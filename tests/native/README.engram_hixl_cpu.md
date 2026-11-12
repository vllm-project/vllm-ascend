# Engram HIXL native CPU harness

This harness includes and executes `csrc/engram_hixl/native.cpp` directly. Its
test-only ACL and HIXL headers replace every external hardware interface; the
build links no Ascend libraries and cannot access a real device. The HIXL stub
records whether initialization happened and which local resource string it
received. A global allocation monitor also checks that a 32 MiB sparse input
does not cause a correspondingly large allocation.

Run from the repository root:

```sh
c++ -std=c++17 -Wall -Wextra -Werror \
  -Itests/native/engram_hixl_cpu_stubs/include \
  tests/native/test_engram_hixl_local_comm_res.cpp \
  -o /tmp/test_engram_hixl_local_comm_res && \
  /tmp/test_engram_hixl_local_comm_res
```

Coverage: empty, 1 byte, exactly 1 MiB, 32 MiB, and mutual exclusion of the
local resource file and listen port. The harness removes its `/tmp` files when
it finishes; no hardware is initialized.
