import json
import os
from dataclasses import dataclass

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.logger import logger

DEFAULT_GLOBAL_SEGMENT_SIZE = 10737418240  # 10 GB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            local_buffer_size=config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)

def mooncake_engine_init(ec_store, mk_config, role):
    if role == ECConnectorRole.SCHEDULER:
        ec_store.setup(
            mk_config.local_hostname,
            mk_config.metadata_server,
            mk_config.global_segment_size,
            mk_config.local_buffer_size,
            "tcp",
            mk_config.device_name,
            mk_config.master_server_address
        )
    else:
        try:
            ec_store.setup(
                mk_config.local_hostname,
                mk_config.metadata_server,
                mk_config.global_segment_size,
                mk_config.local_buffer_size,
                mk_config.protocol,
                mk_config.device_name,
                mk_config.master_server_address
            )
        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise
            
