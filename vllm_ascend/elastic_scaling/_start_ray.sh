#!/bin/sh
HEAD_IP=$(hostname -I | awk '{print $1}')
ray start --head --port=6379 --node-ip-address="$HEAD_IP" \ 
    --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --min-worker-port=10002 --max-worker-port=19999 \
    --resources='{"NPU": 16}'