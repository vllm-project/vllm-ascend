# Support metrics collector component


## Build the component
```bash
cd ./omniifer/omni/accelerators/sched/metrics-collector

sh build.sh
```

## Start metrics collector

```bash
metics-collector -metrics_collector_server="ip:port" -scheduler_server="ip:port" -prefill_servers_list="ip:port,ip:port,..."  -decode_servers_list="ip:port,ip:port,..."
```

Note:

'metrics_collector_server' is your host machine ip and your customize port for your metrics collector.

'scheduler_server' is your global proxy's ip and port.

‘prefill_servers_list’ is your prefill instances' ips and ports.

'decode_servers_list' is your decode instances' ips and ports.