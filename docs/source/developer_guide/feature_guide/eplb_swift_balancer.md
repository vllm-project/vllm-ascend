# Expert Parallelism Load Balancer (EPLB)

## Why we need EPLB?
When using expert parallelism (EP), different experts are assigned to different GPUs/NPUs. Because the load of different experts may vary depending on the current workload, it is important to keep the load of different GPUs/NPUs balanced. We adopt a redundant experts strategy that duplicates heavy-loaded experts. Then, we heuristically pack the duplicated experts to GPUs to ensure load balancing across different GPUs. Moreover, thanks to the group-limited expert routing used in moe model, we also attempt to place the experts of the same group to the same node to reduce inter-node data traffic, whenever possible.

To facilitate reproduction and deployment, we open-source our deployed EP load balancing algorithm in eplb.py. The algorithm computes a balanced expert replication and placement plan based on the estimated expert loads. Note that the exact method to predict the loads of experts is out of this repo's scope. A common method is to use moving average of historical statistics.

![eplb](./images/eplb.png)
## How to use EPLB?
