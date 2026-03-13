# Container Snapshot and Restore (GRUS)

## **Introduction**

When a vLLM inference service is **ready** (model loaded and engine steady), you can **checkpoint** the entire runtime state of the vLLM **container**: weights on disk, NPU runtime state, and host process state. Later, a new pod can **restore** from that snapshot and continue from the checkpoint instead of a full cold start.

This feature is implemented with **GRUS** on Kubernetes. The flow has two layers:

* **Application layer** (`/suspend` / `/resume`): export or reload model weights and NPU state APIs.
* **Platform layer** (GRUS): checkpoint **host-side** process state into a container image.

## **Prerequisites**

* **GRUS** is installed and configured on the cluster. Follow GRUS documentation for version and operator setup.
* **Container runtime**: GRUS requires **containerd** as the Kubernetes CRI (other runtimes are not supported for this workflow).

## **Create a Snapshot**

Complete the steps **in order**: suspend the engine first, then ask GRUS to snapshot the container.

### **1. Suspend the vLLM service**

Call **`POST /suspend`** so the engine dumps **runtime model weights** and backs up **NPU-side** state before the host is frozen.

| Query parameter   | Required | Description |
|-------------------|----------|-------------|
| `model_save_path` | Yes      | Writable path inside the container for the exported weights and dump. |

Example:

```bash
curl -X POST \
  "http://<host>:<port>/suspend?model_save_path=/path/to/model/save"
```

### **2. Create the container snapshot with GRUS**

Use the **GRUS service** (CLI or API provided by your platform) to create a **container snapshot** of the running vLLM pod. This captures **host-side** process memory and related state (implementation details depend on GRUS version).

## **Restore from a Snapshot**

### **1. Configure the workload**

* Set the pod **runtime class** to **grus**.
* Set **GRUS environment variables** (snapshot save path, restore flag, etc.) according to your platform’s checklist.

### **2. Start the container**

Deploy or scale the vLLM workload so the pod starts **from the saved snapshot image**. Wait until the pod is running and the api server is reachable.

### **3. Resume the engine**

Call **`POST /resume`** so the engine reloads **weights**, **NPU state**, and aligns with **host** state after restore.

| Query parameter             | Required | Description |
|----------------------------|----------|-------------|
| `data_parallel_master_ip`  | Yes      | Current DP master IP after reschedule (multi-replica / DP setups). |
| `model_path`               | Yes      | Path to model weights used for restore (must match your layout). |

Example:

```bash
curl -X POST \
  "http://<host>:<port>/resume?data_parallel_master_ip=<MASTER_IP>&model_path=/path/to/model"
```
