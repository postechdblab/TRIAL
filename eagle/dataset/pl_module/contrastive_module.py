import functools
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eagle.dataset import ContrastiveDataset
from eagle.dataset.wrapper import DatasetWrapper
from eagle.dataset.pl_module.base_module import BaseDataModule
from eagle.dataset.utils import (
    preprocess,
    read_compressed,
    read_corpus,
    read_queries,
    save_compressed,
)

logger = logging.getLogger("ContrastiveDataModule")


class ContrastiveDataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig, skip_train: bool = False):
        super().__init__(cfg)
        self.skip_train = skip_train

    def _preprocess_train_data(self, corpus: Dict, queries: Dict) -> Dataset:
        logger.info(
            f"Loading training dataset from {os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.train_contrastive.data_file)}..."
        )
        train_raw_dataset = ContrastiveDataset(
            cfg=self.cfg.train_contrastive,
            cfg_dataset=self.cfg,
            queries=queries,
            override_nway=self.cfg.cache_nway,
        )

        logger.info(f"Dataset loaded! Train dataset size: {len(train_raw_dataset)}")
        logger.info(f"Converting train dataset to HuggingFace format...")
        train_dataset = Dataset.from_dict(train_raw_dataset.to_dict(corpus=corpus))
        logger.info(f"Dataset converted! Train dataset size: {len(train_dataset)}")
        logger.info(f"Preprocessing dataset...")
        train_preprocess_batch = functools.partial(
            preprocess,
            q_tokenizer=self.q_tokenizer,
            d_tokenizer=self.d_tokenizer,
            is_eval=False,
        )
        train_dataset = train_dataset.map(
            train_preprocess_batch,
            batched=True,
            remove_columns=train_raw_dataset.dict_keys,
            desc="Preprocessing train dataset",
        )
        return train_dataset

    def _preprocess_val_data(self, corpus: Dict, queries: Dict) -> Dataset:
        logger.info(
            f"Loading validation dataset from {os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.val.data_file)}..."
        )
        val_raw_dataset = ContrastiveDataset(
            cfg=self.cfg.val,
            cfg_dataset=self.cfg,
            queries=queries,
            override_nway=self.cfg.val.override_nway,
        )
        logger.info(f"Dataset loaded! Eval dataset size: {len(val_raw_dataset)}")
        logger.info(f"Converting val dataset to HuggingFace format...")
        val_dataset = Dataset.from_dict(val_raw_dataset.to_dict(corpus=corpus))
        logger.info(f"Dataset converted! Val dataset size: {len(val_dataset)}")
        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        val_preprocess_batch = functools.partial(
            preprocess,
            q_tokenizer=self.q_tokenizer,
            d_tokenizer=self.d_tokenizer,
            is_eval=True,
        )
        val_dataset = val_dataset.map(
            val_preprocess_batch,
            batched=True,
            remove_columns=val_raw_dataset.dict_keys,
            desc="Preprocessing eval dataset",
        )
        return val_dataset

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        Create cache if not in debug mode.
        """
        # Check if debug mode and skip cache creation
        if self.cfg.is_debug:
            logger.info("Debug mode is enabled. Skipping cache creation...")
            return None

        # Check if cache exists and skip cache creation
        skip_train = os.path.exists(self.training_cache_path) or self.skip_train
        skip_val = os.path.exists(self.validation_cache_path)
        if skip_train and skip_val:
            logger.info(
                f"Cache file exists ({self.training_cache_path}), skipping cache creation..."
            )
            return None

        # Preprocess and create cache
        # Load corpus and query mapping
        if not (skip_train and skip_val):
            corpus = read_corpus(self.corpus_path)
            queries = read_queries(self.queries_path)

        logger.info(f"Loading and preprocessing data to create cache...")
        if not skip_train:
            train_dataset = self._preprocess_train_data(corpus=corpus, queries=queries)
            logger.info(
                f"Saving training dataset (len: {len(train_dataset)}) to {self.training_cache_path}..."
            )
            save_compressed(self.training_cache_path, train_dataset)

        if not skip_val:
            val_dataset = self._preprocess_val_data(corpus=corpus, queries=queries)
            logger.info(
                f"Saving validation dataset (len: {len(val_dataset)}) to {self.validation_cache_path}..."
            )
            save_compressed(self.validation_cache_path, val_dataset)

        logger.info(f"Dataset preprocessed and saved!")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        # Check if cache exists
        if self.cfg.is_debug:
            # Load and preprocess data
            corpus = read_corpus(self.corpus_path)
            queries = read_queries(self.queries_path)
            logger.info(f"Debug mode is enabled. Loading sample data...")
            train_dataset = None
            if not self.skip_train:
                train_dataset = self._preprocess_train_data(
                    corpus=corpus, queries=queries
                )
            val_dataset = self._preprocess_val_data(corpus=corpus, queries=queries)
        else:
            # Load cached training dataset
            train_dataset = None
            if not self.skip_train:
                assert os.path.exists(
                    self.training_cache_path
                ), f"Cache file does not exist: {self.training_cache_path}"
                logger.info(f"Loading dataset from cache {self.training_cache_path}...")
                train_dataset: Dataset = read_compressed(self.training_cache_path)
                logger.info(
                    f"Loaded training dataset (len:{len(train_dataset)}) from cache {self.training_cache_path}"
                )
            # Load cached validation dataset
            assert os.path.exists(
                self.validation_cache_path
            ), f"Cache file does not exist: {self.validation_cache_path}"
            logger.info(f"Loading dataset from cache {self.validation_cache_path}...")
            val_dataset: Dataset = read_compressed(self.validation_cache_path)
            logger.info(
                f"Loaded validation dataset (len:{len(val_dataset)}) from cache {self.validation_cache_path}"
            )

        # Load word and phrase ranges for query and document
        logger.info(f"Loading word and phrase ranges...")
        q_word_ranges = q_phrase_ranges = d_word_ranges = d_phrase_ranges = None
        # Load word ranges
        file_name = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_word_range_file
        )
        if os.path.exists(file_name):
            q_word_ranges = file_utils.read_pickle_file(file_name)
        file_name = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_word_range_file
        )
        if os.path.exists(file_name):
            d_word_ranges = file_utils.read_pickle_file(file_name)
        # Load phrase ranges
        file_name = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_phrase_range_file
        )
        if os.path.exists(file_name):
            q_phrase_ranges = file_utils.read_pickle_file(file_name)
        file_name = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_phrase_range_file
        )
        if os.path.exists(file_name):
            d_phrase_ranges = file_utils.read_pickle_file(file_name)

        # Shuffle data to avoid qid repetition in the mini-batch
        indices = None
        if not self.cfg.is_debug and not self.skip_train:
            indices = self.get_shuffled_indices_to_avoid_qid_repetition(train_dataset)

        # Create DatasetWrapper
        if self.skip_train:
            train_dataset = []
        else:
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
                data_path=self.training_data_path,
                cache_nway=self.cfg.cache_nway,
                q_skip_ids=self.q_tokenizer.special_toks_ids,
                d_skip_ids=self.d_tokenizer.special_toks_ids
                + self.d_tokenizer.punctuations,
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
            data_path=self.validation_data_path,
            cache_nway=self.cfg.val.override_nway,
            q_skip_ids=self.q_tokenizer.special_toks_ids,
            d_skip_ids=self.d_tokenizer.special_toks_ids
            + self.d_tokenizer.punctuations,
            granularity_level=self.cfg_global.model.granularity_level,
            is_use_fine_grained_loss=self.cfg_global.model.is_use_fine_grained_loss,
        )

        # Save datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset

    def train_dataloader(self) -> Union[DataLoader, None]:
        if self.skip_train:
            return None
        return super().train_dataloader()
