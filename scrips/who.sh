#!/bin/bash

PID=$1

if [ -z "$PID" ]; then
  echo "Usage: $0 <PID>"
  exit 1
fi

container_scope=$(cat /proc/$PID/cgroup | grep 'docker' | sed 's/.*\///' | head -n 1)

container_id=$(echo "$container_scope" | sed 's/docker-\(.*\)\.scope/\1/')

if [ -z "$container_id" ]; then
  echo "No container found for PID $PID"
  exit 1
fi

echo "Process $PID is running in container $container_id"
docker ps -a --filter "id=$container_id" --format "table {{.ID}}\t{{.Image}}\t{{.Names}}"