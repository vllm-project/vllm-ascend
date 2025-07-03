#!/bin/bash

HOST_IP=$1
WORKER_PORT=$2

dscli stop \
    --worker_address ${HOST_IP}:${WORKER_PORT}

pkill etcd