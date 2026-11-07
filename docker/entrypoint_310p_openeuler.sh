#!/bin/bash
# No-op on hosts with Ascend accelerators; otherwise start the driver daemons in container mode.
if lspci 2>/dev/null | grep -qi "accelerators"; then
    :
else
    su - HwHiAiUser -c "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:/usr/lib64 && /lib/ld-linux-aarch64.so.1 /var/slogd -d >/dev/null &"
    su - HwDmUser -c "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:/usr/lib64 && /lib/ld-linux-aarch64.so.1 /var/dmp_daemon -I -M -U 8087 &"

    rm -rf /lib64/ld-linux-aarch64.so.1
    ln -sf /lib/ld-linux-aarch64.so.1 /lib64/ld-linux-aarch64.so.1
fi

exec "$@"
