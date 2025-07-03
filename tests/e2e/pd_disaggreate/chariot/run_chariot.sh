#!/bin/bash

HOST_IP=$1
WORKER_PORT=$2
ETCD_PORT=$3

MASTER_PORT=`expr ${WORKER_PORT} + 1`
ETCD_PEER_PORT=`expr ${ETCD_PORT} + 1`

etcd \
    --name etcd-chariot \
    --data-dir /tmp/etcd-chariot \
    --listen-client-urls http://${HOST_IP}:${ETCD_PORT} \
    --advertise-client-urls http://${HOST_IP}:${ETCD_PORT} \
    --listen-peer-urls http://${HOST_IP}:${ETCD_PEER_PORT} \
    --initial-advertise-peer-urls http://${HOST_IP}:${ETCD_PEER_PORT} \
    --initial-cluster etcd-chariot=http://${HOST_IP}:${ETCD_PEER_PORT} &


dscli start \
    -w \
    --worker_address ${HOST_IP}:${WORKER_PORT} \
    --master_address ${HOST_IP}:${MASTER_PORT} \
    --etcd_address ${HOST_IP}:${ETCD_PORT} &