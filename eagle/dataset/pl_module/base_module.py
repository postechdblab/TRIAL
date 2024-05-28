
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
from eagle.dataset.utils import (
    collate_fn, get_indices_to_avoid_repeated_qids_in_minibatch, preprocess,
    read_compressed, read_corpus, read_qrels_qids, read_queries,
    save_compressed)
from eagle.dataset.wrapper import DatasetWrapper
from eagle.tokenizer import DTokenizer, QTokenizer

logger = logging.getLogger("DataModule")


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, skip_train: bool=False):
        super().__init__()
        self.cfg_global: DictConfig = cfg
        self.cfg: DictConfig = cfg.dataset
        self.q_tokenizer = QTokenizer(cfg=cfg.q_tokenizer, model_name=cfg.model.name)
        self.d_tokenizer = DTokenizer(cfg=cfg.d_tokenizer, model_name=cfg.model.name)
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.skip_train = skip_train
        # Check if initializers are valid
        assert len(self.q_tokenizer) == len(
            self.d_tokenizer
        ), f"Tokenizers have different sizes: {len(self.q_tokenizer)} vs {len(self.d_tokenizer)}"

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
      
    def _preprocess_data(self, dataset: BaseDataset, corpus: Dict, is_eval:bool=False) -> Dataset:
        logger.info(f"Converting val dataset to HuggingFace format...")
        val_dataset = Dataset.from_dict(dataset.to_dict(corpus=corpus))
        logger.info(f"Dataset converted! Val dataset size: {len(val_dataset)}")
        
        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        val_preprocess_batch = functools.partial(
            preprocess,
            q_tokenizer=self.q_tokenizer,
            d_tokenizer=self.d_tokenizer,
            is_eval=is_eval,
        )
        val_dataset = val_dataset.map(
            val_preprocess_batch,
            batched=True,
            remove_columns=dataset.dict_keys,
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
            train_dataset = self._preprocess_data(dataset=self._load_train_data(queries=queries), corpus=corpus, is_eval=False)
            logger.info(
                f"Saving training dataset (len: {len(train_dataset)}) to {self.training_cache_path}..."
            )
            save_compressed(self.training_cache_path, train_dataset)

        if not skip_val:
            val_dataset = self._preprocess_data(dataset=self._load_val_data(queries=queries), corpus=corpus, is_eval=True)
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
                train_dataset = self._preprocess_data(dataset=self._load_train_data(queries=queries), corpus=corpus, is_eval=False)
            val_dataset = self._preprocess_data(dataset=self._load_val_data(queries=queries), corpus=corpus, is_eval=True)
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
                cache_nway=self.cfg.cache_nway,
                q_skip_ids=self.q_tokenizer.skip_tok_ids,
                d_skip_ids=self.d_tokenizer.skip_tok_ids,
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
            q_skip_ids=self.q_tokenizer.skip_tok_ids,
            d_skip_ids=self.d_tokenizer.skip_tok_ids,
            granularity_level=self.cfg_global.model.granularity_level,
            is_use_fine_grained_loss=self.cfg_global.model.is_use_fine_grained_loss,
        )

        # Save datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset

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

    @abc.abstractmethod
    def _load_train_data(self, queries: Dict) -> Dataset:
        raise NotImplementedError
        
    @abc.abstractmethod
    def _load_val_data(self, queries: Dict) -> Dataset:
        raise NotImplementedError