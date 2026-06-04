import json
import math
import os
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


def load_seq(x: str):
    if isinstance(x, str):
        y = json.loads(x)
    else:
        y = x
    return y


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = True,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return loader


class InferenceDataset(IterableDataset):
    """
    SequenceDataset is an iterable dataset designed for distributed recommendation systems.
    It handles loading, shuffling, and batching of sequence data for training models.

    Args:
        seq_logs_file (str): Path to the sequence logs file.
        batch_size (int): The batch size.
        max_seqlen (int): The maximum sequence length.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (list[str], optional): List of contextual feature names. Defaults to [].
        action_feature_name (str, optional): The name of the action feature. Defaults to None.
        max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        shuffle (bool): Whether to shuffle the data.
        random_seed (int): The random seed for shuffling.
        is_train_dataset (bool): Whether this dataset is for training.
        nrows (int, optional): The number of rows to read from the file. Defaults to None, meaning all rows are read.
    """

    def __init__(
        self,
        seq_logs_file: str,
        batch_logs_file: str,
        batch_size: int,
        max_seqlen: int,
        item_feature_name: str,
        contextual_feature_names: List[str],
        action_feature_name: str,
        max_num_candidates: int = 0,
        *,
        item_vocab_size: int,
        userid_name: str,
        date_name: str,
        sequence_endptr_name: str,
        timestamp_names: List[str],
        random_seed: int = 0,
        seq_nrows: Optional[int] = None,
        batch_nrows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._device = torch.device("cpu")

        if seq_nrows is None or batch_nrows is None:
            self._seq_logs_frame = pd.read_csv(seq_logs_file, delimiter=",")
            self._batch_logs_frame = pd.read_csv(batch_logs_file,
                                                 delimiter=",")
        else:
            self._seq_logs_frame = pd.read_csv(
                seq_logs_file, delimiter=",",
                nrows=seq_nrows)  # 加载processed_seqs.csv
            self._batch_logs_frame = pd.read_csv(
                batch_logs_file, delimiter=",",
                nrows=batch_nrows)  # 加载processed_batches.csv

        self._batch_logs_frame.sort_values(by=[userid_name, date_name],
                                           inplace=True)
        self._seq_logs_frame.sort_values(by=[userid_name, date_name],
                                         inplace=True)
        self._num_samples = len(self._batch_logs_frame)
        self._num_seq_samples = len(self._seq_logs_frame)
        self._max_seqlen = max_seqlen

        self._batch_size = batch_size
        self._random_seed = random_seed

        self._contextual_feature_names = contextual_feature_names
        if self._max_seqlen <= len(self._contextual_feature_names):
            raise ValueError(
                f"max_seqlen is too small. should > {len(self._contextual_feature_names)}"
            )
        self._item_feature_name = item_feature_name
        self._action_feature_name = action_feature_name
        self._max_num_candidates = max_num_candidates
        self._item_vocab_size = item_vocab_size
        self._userid_name = userid_name
        self._date_name = date_name
        self._seq_end_name = sequence_endptr_name

        self._sample_ids = np.arange(self._num_samples)
        self._sample_seq_ids = np.arange(self._num_seq_samples)

    # We do batching in our own
    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            batch_start = (i * self._batch_size)
            batch_end = min(
                (i + 1) * self._batch_size,
                len(self._sample_seq_ids),
            )
            sample_ids = self._sample_seq_ids[batch_start:batch_end]
            user_ids: List[int] = []
            dates: List[int] = []
            action_weights: List[int] = []
            video_ids: List[int] = []
            for sample_id in sample_ids:
                user_ids.append(
                    self._seq_logs_frame.iloc[sample_id][self._userid_name])
                dates.append(
                    self._seq_logs_frame.iloc[sample_id][self._date_name])
                video_ids += (load_seq(self._seq_logs_frame.iloc[sample_id][
                    self._item_feature_name]))
                action_weights += (load_seq(
                    self._seq_logs_frame.iloc[sample_id][
                        self._action_feature_name]))
            yield (user_ids, dates, video_ids, action_weights)
