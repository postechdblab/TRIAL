import abc
import functools
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import lightning as L
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.pl_module.utils import tokenize_and_cache_corpus
from eagle.dataset.utils import (
    collate_fn,
    get_indices_to_avoid_repeated_qids_in_minibatch,
    preprocess,
    read_compressed,
    read_corpus,
    read_qrels_qids,
    read_queries,
    save_compressed,
)
from eagle.dataset.wrapper import DatasetWrapper
from eagle.tokenizer import Tokenizers

logger = logging.getLogger("DataModule")


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, skip_train: bool = False):
        super().__init__()
        self.cfg_global: DictConfig = cfg
        self.cfg: DictConfig = cfg.dataset
        self.tokenizers = Tokenizers(cfg.q_tokenizer, cfg.d_tokenizer, cfg.model.name)
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.skip_train = skip_train

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.corpus_file)

    @property
    def queries_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.query_file)

    @property
    def corpus_cache_path(self) -> str:
        return self.corpus_path + ".tok.cache"

    @property
    def queries_cache_path(self) -> str:
        return self.queries_path + ".tok.cache"

    @property
    def q_word_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_word_range_file
        )

    @property
    def q_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_phrase_range_file
        )

    @property
    def d_word_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_word_range_file
        )

    @property
    def d_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_phrase_range_file
        )

    @functools.cached_property
    def corpus_mapping(self) -> Dict[str, int]:
        path = self.corpus_path
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            collection: Dict[str, str] = read_corpus(path)
            mapping = {key: idx for idx, key in enumerate(collection.keys())}
            save_compressed(cache_path, mapping)
        return mapping

    @functools.cached_property
    def query_mapping(self) -> Dict[str, int]:
        path = self.queries_path
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            queries = file_utils.read_json_file(path, auto_detect_extension=True)
            mapping = {str(item["_id"]): idx for idx, item in enumerate(queries)}
            save_compressed(cache_path, mapping)
        return mapping

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        Create cache if not in debug mode.
        """
        # Create a tokenized cache for entire document corpus
        if not os.path.exists(self.corpus_cache_path):
            logger.info(f"Reading all documents...")
            d_corpus = read_corpus(self.corpus_path)
            logger.info("Tokenizing and caching document corpus...")
            d_items = tokenize_and_cache_corpus(
                tokenizer=self.tokenizers.d_tokenizer, corpus=d_corpus
            )
            # Write the cache
            logger.info(
                f"Saving {len(d_items)} tokenized document cache to {self.corpus_cache_path}"
            )
            save_compressed(self.corpus_cache_path, d_items)

        # Create a tokenized cache for entire query set
        if not os.path.exists(self.queries_cache_path):
            logger.info("Reading all queries...")
            q_corpus = read_queries(self.queries_path)
            logger.info("Tokenizing and caching query set...")
            q_items = tokenize_and_cache_corpus(
                tokenizer=self.tokenizers.d_tokenizer, corpus=q_corpus
            )
            # Write the cache
            logger.info(
                f"Saving {len(q_items)} tokenized query cache to {self.queries_cache_path}"
            )
            save_compressed(self.queries_cache_path, q_items)

        logger.info(f"Dataset preprocessed and saved!")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        # Load tokenized corpus and queries
        logger.info(f"Loading tokenized corpus and queries...")
        tokenized_corpus = read_compressed(self.corpus_cache_path)
        tokenized_queries = read_compressed(self.queries_cache_path)

        # Load train and val data
        train_dataset = []
        if not self.skip_train:
            train_dataset: BaseDataset = self._load_train_data(
                tokenized_queries=tokenized_queries, tokenized_corpus=tokenized_corpus
            )
        val_dataset: BaseDataset = self._load_val_data(
            tokenized_queries=tokenized_queries, tokenized_corpus=tokenized_corpus
        )

        # Load word and phrase ranges for query and document
        logger.info(f"Loading word and phrase ranges...")
        q_word_ranges = q_phrase_ranges = d_word_ranges = d_phrase_ranges = None
        # Load word ranges
        need_to_load_ranges = self.cfg_global.model.granularity_level != "token"
        if need_to_load_ranges and os.path.exists(self.q_word_range_path):
            q_word_ranges = file_utils.read_pickle_file(self.q_word_range_path)
        if need_to_load_ranges and os.path.exists(self.d_word_range_path):
            d_word_ranges = file_utils.read_pickle_file(self.d_word_range_path)
        # Load phrase ranges
        if need_to_load_ranges and os.path.exists(self.q_phrase_range_path):
            q_phrase_ranges = file_utils.read_pickle_file(self.q_phrase_range_path)
        if need_to_load_ranges and os.path.exists(self.d_phrase_range_path):
            d_phrase_ranges = file_utils.read_pickle_file(self.d_phrase_range_path)

        # Shuffle data to avoid qid repetition in the mini-batch
        indices = None
        if not self.cfg.is_debug and not self.skip_train:
            indices = self.get_shuffled_indices_to_avoid_qid_repetition(train_dataset)

        # Create DatasetWrapper
        if not self.skip_train:
            train_dataset = DatasetWrapper(
                dataset=train_dataset,
                indices=indices,
                q_word_ranges=q_word_ranges,
                q_phrase_ranges=q_phrase_ranges,
                d_word_ranges=d_word_ranges,
                d_phrase_ranges=d_phrase_ranges,
                corpus_mapping=self.corpus_mapping,
                query_mapping=self.query_mapping,
                nway=self.cfg_global.training.nway,
                cache_nway=self.cfg.cache_nway,
                q_skip_ids=self.tokenizers.q_tokenizer.skip_tok_ids,
                d_skip_ids=self.tokenizers.d_tokenizer.skip_tok_ids,
                granularity_level=self.cfg_global.model.granularity_level,
                is_use_fine_grained_loss=self.cfg_global.model.is_use_fine_grained_loss,
            )

        val_dataset = DatasetWrapper(
            dataset=val_dataset,
            q_word_ranges=q_word_ranges,
            q_phrase_ranges=q_phrase_ranges,
            d_word_ranges=d_word_ranges,
            d_phrase_ranges=d_phrase_ranges,
            corpus_mapping=self.corpus_mapping,
            query_mapping=self.query_mapping,
            nway=self.cfg.val.override_nway,
            cache_nway=self.cfg.val.override_nway,
            q_skip_ids=self.tokenizers.q_tokenizer.skip_tok_ids,
            d_skip_ids=self.tokenizers.d_tokenizer.skip_tok_ids,
            granularity_level=self.cfg_global.model.granularity_level,
            is_use_fine_grained_loss=self.cfg_global.model.is_use_fine_grained_loss,
        )

        # Save datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset

    def get_shuffled_indices_to_avoid_qid_repetition(
        self, train_dataset: BaseDataset
    ) -> List[int]:
        """Make sure that there are no repeated qids in the batch.
        So that in-batch negatives are valid."""
        # Load training qrels to avoid loading the entire training dataset (instead, we stream it during training)
        train_qids: List[str] = read_qrels_qids(self.train_qrels_path)
        # Check the loaded data is valid
        assert len(train_qids) == len(
            train_dataset
        ), f"len(qids)={len(train_qids)}, len(train_dataset)={len(train_dataset)}"

        # Get new indices to avoid repeated qids in the mini-batch
        bsize = self.cfg_global.training.per_device_train_batch_size
        indices: List[int] = get_indices_to_avoid_repeated_qids_in_minibatch(
            train_qids, bsize
        )
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

    def train_dataloader(self) -> Union[DataLoader, None]:
        if self.skip_train:
            return None
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

    @property
    @abc.abstractmethod
    def train_qrels_path(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _load_train_data(
        self, tokenized_queries: Dict, tokenized_corpus: Dict
    ) -> Union[BaseDataset, None]:
        raise NotImplementedError

    @abc.abstractmethod
    def _load_val_data(
        self, tokenized_queries: Dict, tokenized_corpus: Dict
    ) -> BaseDataset:
        raise NotImplementedError
