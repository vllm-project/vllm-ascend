#!/usr/bin/env python3

import argparse
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import List, Dict

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default OpenLibing upload URL
DEFAULT_UPLOAD_URL = "https://apig.openlibing.com/openlibing-sync/sync/testcase/metadata/upload"


def _process_json_param(param: str) -> Dict:
    """
    Process JSON parameter that may be in simplified format (without quotes).
    
    Supports both:
    - Standard JSON: {"key": "value", ...}
    - Simplified format: {key: value, ...} or {key:value,key:value}
    
    Args:
        param: Input string to process
        
    Returns:
        Dict: Parsed JSON dictionary, empty dict if parsing fails
    """
    try:
        if param and param.find('"') == -1:
            inner = param.strip('{}')
            pairs = [p.strip() for p in inner.split(',')]
            json_pairs = []
            for pair in pairs:
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    json_pairs.append(f'"{k.strip()}":"{v.strip()}"')
            tmp_secret = '{' + ','.join(json_pairs) + '}'
            param: dict = json.loads(tmp_secret)
        else:
            param: dict = json.loads(param)
    except Exception as e:
        logger.info(f"process special json err: {e}", exc_info=True)
        param = {}

    return param


def upload_data_to_openlibing(
    file_paths: List[Path],
    openlibing_secret: Dict[str, str],
    upload_config: Dict[str, str] = None
) -> requests.Response:
    """
    Upload files to OpenLibing OBS bucket.

    Args:
        file_paths: List of file paths to upload
        openlibing_secret: Dictionary containing apig_code, apig_key, apig_secret
        upload_config: Configuration dictionary containing:
            - pipeline_id: Pipeline ID (optional)
            - pipeline_run_id: Pipeline run ID (optional)
            - job_id: Job ID (optional)
            - url: Upload URL (default: OpenLibing production URL)
            - label: Optional label string for identifying this upload
            - archive_path: Custom archive path, must be used with label

    Returns:
        requests.Response: The HTTP response object from the upload request

    Raises:
        FileNotFoundError: If no valid files to upload
        Exception: If upload fails
    """
    upload_config = upload_config or {}

    pipeline_id = upload_config.get("pipeline_id", "")
    pipeline_run_id = upload_config.get("pipeline_run_id", "")
    job_id = upload_config.get("job_id", "")
    url = upload_config.get("url", DEFAULT_UPLOAD_URL)
    label = upload_config.get("label", "")
    archive_path = upload_config.get("archive_path", "")
    label_info = f" (label: {label})" if label else ""
    logger.info(f"Uploading {len(file_paths)} files to OpenLibing{label_info}")
    
    if not file_paths:
        raise FileNotFoundError("No files to upload")

    headers = {
        "X-Apig-Appcode": openlibing_secret.get("apig_code", ""),
        "AppKey": openlibing_secret.get("apig_key", ""),
        "AppSecret": openlibing_secret.get("apig_secret", ""),
        "User-Agent": "Python-requests/2.25.0",
    }

    form_data = {}
    if pipeline_id:
        form_data["pipelineId"] = pipeline_id
    if pipeline_run_id:
        form_data["pipelineRunId"] = pipeline_run_id
    if job_id:
        form_data["jobId"] = job_id

    archive_config = {}
    if label:
        archive_config["label"] = label
    if archive_path:
        archive_config["archivePath"] = archive_path
    if archive_config:
        form_data["archiveConfig"] = json.dumps(archive_config)

    opened_files = []
    files_for_upload = []

    try:
        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File not found, skipping: {file_path}")
                continue

            f = open(file_path, 'rb')
            opened_files.append(f)

            mime_type = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'

            files_for_upload.append((
                'files',
                (file_path.name, f, mime_type)
            ))

        if not files_for_upload:
            raise FileNotFoundError("No valid files to upload")

        logger.info(f"Uploading {len(files_for_upload)} files to {url}")
        
        response = requests.post(
            url=url,
            headers=headers,
            data=form_data,
            files=files_for_upload,
            verify=False,
        )

        logger.info(f"Upload response status: {response.status_code}")
        logger.info(f"Upload response text: {response.text}")

        response.raise_for_status()
        
        logger.info("Successfully uploaded files to OpenLibing")
        
        return response

    except Exception as e:
        logger.error(f"Failed to upload files to OpenLibing: {e}", exc_info=True)
        raise

    finally:
        for f in opened_files:
            try:
                f.close()
            except Exception as close_err:
                logger.warning(f"Failed to close file: {close_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload files to OpenLibing OBS bucket",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--files",
        nargs='+',
        required=True,
        type=Path,
        help="List of file paths to upload (supports any file type)"
    )
    
    parser.add_argument(
        "--pipeline-id",
        default="",
        help="Pipeline ID (optional)"
    )
    
    parser.add_argument(
        "--pipeline-run-id",
        default="",
        help="Pipeline run ID (optional)"
    )
    
    parser.add_argument(
        "--job-id",
        default="",
        help="Job ID (optional)"
    )
    
    parser.add_argument(
        "--label",
        default="",
        help="Label string. Can be used alone (archive path: /{label}/{filename}) "
             "or with --archive-path (archive path: /{label}/{archive_path}/{filename})."
    )
    
    parser.add_argument(
        "--archive-path",
        default="",
        help="Custom archive path, must be used with --label. "
             "Archive path becomes /{label}/{archive_path}/{filename}. "
             "Can be used together with --pipeline-id/--pipeline-run-id/--job-id, "
             "in which case --archive-path takes precedence. "
             "Must provide at least --label or pipeline params."
    )
    
    parser.add_argument(
        "--openlibing-secret",
        default="",
        help="JSON string containing apig_code, apig_key, apig_secret. "
             "Supports both standard JSON format {\"key\": \"value\"} and "
             "simplified format {key: value} without quotes. "
             "If not provided, reads from environment variable OPENLIBING_SECRET. "
             "Errors if neither is available."
    )
    
    parser.add_argument(
        "--url",
        default=DEFAULT_UPLOAD_URL,
        help=f"Upload URL (default: {DEFAULT_UPLOAD_URL})"
    )
    
    args = parser.parse_args()
    
    try:
        has_archive_path = bool(args.archive_path)
        has_label = bool(args.label)
        has_pipeline_params = bool(args.pipeline_id or args.pipeline_run_id or args.job_id)

        if has_archive_path and not has_label:
            parser.error("--archive-path requires --label")
        if not has_label and not has_pipeline_params:
            parser.error("must provide either --label or --pipeline-id/--pipeline-run-id/--job-id")

        secret_raw = args.openlibing_secret or os.environ.get("OPENLIBING_SECRET", "")
        if not secret_raw:
            logger.error("openlibing-secret not provided via --openlibing-secret or OPENLIBING_SECRET env var")
            exit(1)

        openlibing_secret = _process_json_param(secret_raw)
        
        if not openlibing_secret:
            logger.error("Failed to parse openlibing-secret")
            exit(1)
            
        response = upload_data_to_openlibing(
            file_paths=args.files,
            openlibing_secret=openlibing_secret,
            upload_config={
                "pipeline_id": args.pipeline_id,
                "pipeline_run_id": args.pipeline_run_id,
                "job_id": args.job_id,
                "url": args.url,
                "label": args.label,
                "archive_path": args.archive_path
            }
        )
        
        logger.info("=== Upload Response Details ===")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        logger.info(f"Response Text: {response.text}")
        logger.info("===============================")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
