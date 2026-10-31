# Deployment Guide

## Troubleshooting

### The engine fails to start with "Address already in use"

In a data parallel deployment, the master binds the data parallel RPC port
(`--dp-rpc-port`, `12321` by default) for the startup handshake. When a
previous engine hangs or exits abnormally, that stale process keeps the port
and the NPU devices alive, so starting an engine again in the same container
fails with:

```text
zmq.error.ZMQError: Address already in use (addr='tcp://<node-ip>:12321')
```

vLLM Ascend checks the port before the engine starts and reports the process
that holds it:

```text
The data parallel RPC port 192.168.0.1:12321 is already in use by PID 1234 (python).
```

Recover from it by stopping the stale process, or by restarting the container
on that node:

```bash
# Show the process that still holds the port.
ss -lntp | grep 12321

# Stop the stale engine process.
kill -9 <pid>
```

Restarting the container always releases both the port and the NPU devices.
