# mdaemon

Python package for the multimodel daemon monitor CLI.

## Install

From repo root:

- `python3 -m pip install -e ./multimodel/daemon`

## Run

- `mdaemon --pool 0:16:4 --pool 1:16:4`
- `mdaemon --pool 0:16:4 --control-enable`

Pool format: `device_id:granularity_mb:total_gb`.

## Control API (localhost)

Monitor can expose a local HTTP API for external orchestrators:

- `--control-enable`
- `--control-host` (default `127.0.0.1`)
- `--control-port` (default `18080`)

Endpoints:

- `GET /healthz`
- `GET /v1/pools`
- `POST /v1/pools/create`
- `POST /v1/pools/extend`
- `POST /v1/pools/remove`

Write request body keys:

- `create`: `device_id`, `granularity`, `total_bytes` (optional `cap_bytes`)
- `extend/remove`: `device_id`, `granularity`, `target_bytes`

`target_bytes` is translated to handle count inside monitor.

Example:

- `curl -s http://127.0.0.1:18080/v1/pools`
