/* Recreate of the local ~/bin/mlock-unlock helper: raise RLIMIT_MEMLOCK to
 * unlimited, then exec the command. The native pin backend mlocks registered
 * ranges and aborts past the default 64 MiB cap.
 *
 * When the container shell cannot raise the limit itself, give this binary
 * file capabilities so the raise works on exec even without them in the
 * shell's effective set:
 *     setcap 'cap_sys_resource,cap_ipc_lock+ep' mlock-unlock
 * (file caps are still bounded by the container's capability bounding set.) */
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
int main(int argc, char** argv) {
  struct rlimit rl = {RLIM_INFINITY, RLIM_INFINITY};
  if (setrlimit(RLIMIT_MEMLOCK, &rl)) {
    perror("setrlimit");
    return 1;
  }
  if (argc > 1) execvp(argv[1], argv + 1);
  perror("exec");
  return 1;
}
