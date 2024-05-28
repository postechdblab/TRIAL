import abc
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eagle.dataset.utils import (
    collate_fn, get_indices_to_avoid_repeated_qids_in_minibatch,
    read_compressed, read_corpus, read_qrels_qids, save_compressed)
from eagle.tokenizer import NewTokenizer

logger = logging.getLogger("DataModule")


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg_global: DictConfig = cfg
        self.cfg: DictConfig = cfg.dataset
        self.q_tokenizer = NewTokenizer(cfg=cfg.q_tokenizer, model_name=cfg.model.name)
        self.d_tokenizer = NewTokenizer(cfg=cfg.d_tokenizer, model_name=cfg.model.name)
        self.train_dataset = self.val_dataset = self.test_dataset = None
        # Check if initializers are valid
        assert len(self.q_tokenizer) == len(
            self.d_tokenizer
        ), f"Tokenizers have different sizes: {len(self.q_tokenizer)} vs {len(self.d_tokenizer)}"

    @property
    def training_data_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.train_contrastive.data_file)

    @property
    def validation_data_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.val.data_file)

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.corpus_file)

    @property
    def queries_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.query_file)

    @property
    def train_qrels_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.train_contrastive.qrel_file)

    @property
    def corpus_mapping(self) -> Dict[str, int]:
        if hasattr(self, "_corpus_mapping"):
            return self._corpus_mapping
        path = self.corpus_path
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            collection: Dict[str, str] = read_corpus(path)
            mapping = {key: idx for idx, key in enumerate(collection.keys())}
            save_compressed(cache_path, mapping)
        # Save mapping to the instance
        self._corpus_mapping = mapping
        return self._corpus_mapping

    @property
    def query_mapping(self) -> Dict[str, int]:
        if hasattr(self, "_query_mapping"):
            return self._query_mapping
        path = self.queries_path
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            queries = file_utils.read_json_file(path, auto_detect_extension=True)
            mapping = {str(item["_id"]): idx for idx, item in enumerate(queries)}
            save_compressed(cache_path, mapping)
        # Save mapping to the instance
        self._query_mapping = mapping
        return self._query_mapping

    @property
    def training_cache_path(self) -> str:
        tokenizer_name_prefix = self.d_tokenizer.name.split("/")[-1].split("-")[0]
        suffix = f"train_dataset.{tokenizer_name_prefix}.{type(self).__name__}.cache"
        data_cache_file_path = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.data_cache_file
        )
        return data_cache_file_path.replace("dataset.cache", suffix)

    @property
    def validation_cache_path(self) -> str:
        tokenizer_name_prefix = self.d_tokenizer.name.split("/")[-1].split("-")[0]
        data_cache_file_path = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.data_cache_file
        )
        suffix = f"val_dataset.{tokenizer_name_prefix}.{type(self).__name__}.cache"
        return data_cache_file_path.replace("dataset.cache", suffix)

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        Create cache if not in debug mode.
        """
        pass
    
    @abc.abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        pass

    def get_shuffled_indices_to_avoid_qid_repetition(self, train_dataset: List[Dict]) -> List[int]:
        """Make sure that there are no repeated qids in the batch.
        So that in-batch negatives are valid."""
        # Load training qrels to avoid loading the entire training dataset (instead, we stream it during training)
        train_qids: List[str] = read_qrels_qids(self.train_qrels_path)
        # Check the loaded data is valid
        assert len(train_qids) == len(
            train_dataset
        ), f"len(qids)={len(train_qids)}, len(train_dataset)={len(train_dataset)}"
        assert (
            train_qids[0] == train_dataset[0]["q_id"]
        ), f"qids[0]={train_qids[0]}, train_dataset[0]['q_id']={train_dataset[0]['q_id']}"
        assert (
            train_qids[-1] == train_dataset[-1]["q_id"]
        ), f"qids[-1]={train_qids[-1]}, train_dataset[-1]['q_id']={train_dataset[-1]['q_id']}"

        # Get new indices to avoid repeated qids in the mini-batch
        bsize = self.cfg_global.training.per_device_train_batch_size
        indices: List[int] = get_indices_to_avoid_repeated_qids_in_minibatch(train_qids, bsize)
        assert len(indices) == len(
            train_qids
        ), f"len(indices)={len(indices)}, len(train_qids)={len(train_qids)}, bsize={bsize}"

        # Check if there are repeated qids in the mini-batch
        repeated_mini_batch_indices = []
        for i in range(0, len(train_qids), bsize):
            if i + bsize > len(train_qids):
                break
            qids = [train_qids[indices[j]] for j in range(i, i + bsize)]
            if len(qids) != len(set(qids)):
                repeated_mini_batch_indices.append(i)
        if repeated_mini_batch_indices:
            logger.info(
                f"Num of mini-batch with repeated qids: {len(repeated_mini_batch_indices)}/{len(train_qids)//bsize}."
            )
            assert (
                len(repeated_mini_batch_indices) < 10
            ), f"len(repeated_mini_batch_indices)={len(repeated_mini_batch_indices)}"
        else:
            logger.info(f"No repeated qids in the mini-batch.")

        return indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg_global.training.per_device_train_batch_size,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=2,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=2,
            collate_fn=collate_fn,
        )
