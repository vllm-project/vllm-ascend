"""Shared OBS transfer and storage utilities for profiling artifacts."""

import threading
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_BUCKET = "ascend-profile-artifacts"
DEFAULT_ENDPOINT = "https://obs.cn-southwest-2.myhuaweicloud.com"
DEFAULT_REGION = "cn-southwest-2"
GZIP_COMPRESSION_LEVEL = 1


class ArtifactError(RuntimeError):
    """An artifact validation failure whose message is safe to publish."""


# =============================================================================
# Local and OBS transfer helpers
# =============================================================================


def create_obs_client(endpoint: str, region: str):
    """Create the S3-compatible client used to access Huawei OBS."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "virtual", "payload_signing_enabled": False},
        ),
    )


def create_transfer_config():
    """Return bounded multipart-transfer settings shared by uploads/downloads."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=64 * 1024**2,
        multipart_chunksize=64 * 1024**2,
        max_concurrency=4,
        use_threads=True,
    )


def make_transfer_progress_callback(label: str, total_bytes: int):
    """Build a thread-safe callback that reports transfer progress by quarter."""
    transferred = 0
    reported = 0
    lock = threading.Lock()

    def callback(amount: int) -> None:
        """Accumulate one transfer chunk and print newly crossed quarters."""
        nonlocal transferred, reported
        with lock:
            transferred += amount
            percent = min(100, transferred * 100 // total_bytes)
            step = percent // 25
            if step > reported:
                reported = step
                print(f"[Profiling] {label}: {percent}% ({transferred}/{total_bytes} bytes)", flush=True)

    return callback


def upload_archive(
    client,
    transfer_config,
    root: Path,
    prefix: str,
    bucket: str,
    target_record: dict,
    stage: str,
) -> None:
    """Upload one archive and record its location after the SDK reports success."""
    archive_path = root / target_record["archive"]
    object_key = f"{prefix}/{target_record['archive']}"
    size_bytes = archive_path.stat().st_size
    print(f"[Profiling] Upload {stage} {target_record['archive']}: {size_bytes} bytes", flush=True)
    client.upload_file(
        str(archive_path),
        bucket,
        object_key,
        ExtraArgs={"ContentType": "application/gzip"},
        Config=transfer_config,
        Callback=make_transfer_progress_callback(f"Upload {stage} {target_record['archive']}", size_bytes),
    )
    target_record["obs_url"] = f"obs://{bucket}/{object_key}"


def obs_http_url(obs_url: str, endpoint: str) -> str:
    """Convert an OBS object URI to an unsigned, virtual-hosted HTTP(S) URL."""
    bucket, key = obs_url.removeprefix("obs://").split("/", 1)
    server = urlsplit(endpoint)
    return f"{server.scheme}://{bucket}.{server.netloc}/{quote(key, safe='/')}"
