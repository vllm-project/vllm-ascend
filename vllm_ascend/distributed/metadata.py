# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import torch
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDConnectorData


@dataclass
class NPUP2PAFDConnectorMetadata(AFDConnectorData):
    def __init__(self):
        self.handle = None
