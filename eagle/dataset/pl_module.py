import functools
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import lightning as L
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eagle.dataset import RawDataset
from eagle.dataset.dataset_wrapper import DatasetWrapper
from eagle.dataset.utils import (
    collate_fn,
    get_indices_to_avoid_repeated_qids_in_minibatch,
    preprocess,
    read_compressed,
    read_corpus,
    save_compressed,
)
from eagle.tokenizer import NewTokenizer

logger = logging.getLogger("DataModule")


class NewDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, skip_train: bool = False):
        super().__init__()
        self.cfg_global = cfg
        self.cfg = cfg.dataset
        self.cfg_training = cfg.training
        self.skip_train = skip_train
        self.q_tokenizer = NewTokenizer(cfg=cfg.q_tokenizer, model_name=cfg.model.name)
        self.d_tokenizer = NewTokenizer(cfg=cfg.d_tokenizer, model_name=cfg.model.name)
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.corpus_mapping: Dict[str, int] = self._get_corpus_mapping(self.corpus_path)
        self.query_mapping: Dict[str, int] = self._get_queries_mapping(
            self.query_mapping_path
        )
        # Check if initializers are valid
        assert len(self.q_tokenizer) == len(
            self.d_tokenizer
        ), f"Tokenizers have different sizes: {len(self.q_tokenizer)} vs {len(self.d_tokenizer)}"

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.corpus_file)

    @property
    def query_mapping_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.query_file)

    @property
    def training_cache_path(self) -> str:
        tokenizer_name_prefix = self.d_tokenizer.name.split("-")[0]
        if self.cfg_training.is_use_distillation:
            suffix = f"train_dataset.{tokenizer_name_prefix}.distillation.cache"
        else:
            suffix = f"train_dataset.{tokenizer_name_prefix}.cache"
        data_cache_file_path = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.data_cache_file
        )
        return data_cache_file_path.replace("dataset.cache", suffix)

    @property
    def validation_cache_path(self) -> str:
        tokenizer_name_prefix = self.d_tokenizer.name.split("-")[0]
        data_cache_file_path = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.data_cache_file
        )
        return data_cache_file_path.replace(
            "dataset.cache", f"val_dataset.{tokenizer_name_prefix}.cache"
        )

    def _get_corpus_mapping(self, path: str) -> Dict[str, int]:
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            collection: Dict[str, str] = read_corpus(path)
            mapping = {key: idx for idx, key in enumerate(collection.keys())}
            save_compressed(cache_path, mapping)
        return mapping

    def _get_queries_mapping(self, path: str) -> Dict[str, int]:
        assert path.endswith(".jsonl"), f"path={path}"
        cache_path = path + ".mapping.cache"
        if os.path.exists(cache_path):
            mapping = read_compressed(cache_path)
        else:
            queries = file_utils.read_json_file(path, auto_detect_extension=True)
            mapping = {str(item["_id"]): idx for idx, item in enumerate(queries)}
            save_compressed(cache_path, mapping)
        return mapping

    def _load_and_preprocess_data(
        self, skip_train: bool = False, skip_val: bool = False
    ) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        assert (
            not skip_train or not skip_val
        ), f"skip_train={skip_train}, skip_val={skip_val}"
        logger.info(f"Loading collection from {self.corpus_path}...")
        corpus = read_corpus(self.corpus_path)
        logger.info(f"Corpus loaded! Collection size: {len(corpus)}")

        # Creating training dataset
        if skip_train:
            train_dataset = None
            queries_data = (
                self.cfg.query_file
            )  # For efficient loading (which is required both in train and val)
        else:
            logger.info(
                f"Loading training dataset from {os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.train.data_file)}..."
            )
            train_raw_dataset = RawDataset(
                self.cfg.train,
                dir_path=self.cfg.dir_path,
                dataset_name=self.cfg.name,
                queries=self.cfg.query_file,
                override_nway=self.cfg.cache_nway,
                is_use_distillation=self.cfg_training.is_use_distillation,
            )
            queries_data = train_raw_dataset.queries

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
                remove_columns=train_raw_dataset.dict_keys(),
                desc="Preprocessing train dataset",
            )

        # Creating validation dataset
        if skip_val:
            val_dataset = None
        else:
            logger.info(
                f"Loading validation dataset from {os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.val.data_file)}..."
            )
            val_raw_dataset = RawDataset(
                self.cfg.val,
                dir_path=self.cfg.dir_path,
                dataset_name=self.cfg.name,
                queries=queries_data,
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
                remove_columns=val_raw_dataset.dict_keys(),
                desc="Preprocessing eval dataset",
            )

        train_len = len(train_dataset) if train_dataset is not None else 0
        val_len = len(val_dataset) if val_dataset is not None else 0
        logger.info(
            f"Data loaded and preprocessed! Train dataset size: {train_len}, Val dataset size: {val_len}"
        )

        return train_dataset, val_dataset

    def _shuffle_data_to_avoid_qid_repetition(
        self, train_dataset: List[Dict]
    ) -> List[int]:
        """Make sure that there are no repeated qids in the batch.
        So that in-batch negatives are valid."""
        # Load training qrels
        qrels = file_utils.read_csv_file(
            os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.train.qrel_file),
            delimiter="\t",
            first_row_as_header=True,
        )
        train_qids = [item["query-id"] for item in qrels]
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
        bsize = self.cfg_training.per_device_train_batch_size
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

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        Create cache if not in debug mode.
        """
        # Check if cache exists
        if self.cfg.is_debug:
            logger.info("Debug mode is enabled. Skipping cache creation...")
        else:
            if os.path.exists(self.training_cache_path) and os.path.exists(
                self.validation_cache_path
            ):
                logger.info(
                    f"Cache file exists ({self.training_cache_path}), skipping cache creation..."
                )
            else:
                logger.info(f"Loading and preprocessing data to create cache...")
                skip_train = os.path.exists(self.training_cache_path) or self.skip_train
                skip_val = os.path.exists(self.validation_cache_path)
                if skip_train and skip_val:
                    logger.info(
                        f"Cache files exist ({self.validation_cache_path}). Skipping cache creation..."
                    )
                    return None
                train_dataset, val_dataset = self._load_and_preprocess_data(
                    skip_train=skip_train, skip_val=skip_val
                )
                # Save cache
                if train_dataset is not None:
                    logger.info(
                        f"Saving training dataset (len: {len(train_dataset)}) to {self.training_cache_path}..."
                    )
                    save_compressed(self.training_cache_path, train_dataset)
                if val_dataset is not None:
                    logger.info(
                        f"Saving validation dataset (len: {len(val_dataset)}) to {self.validation_cache_path}..."
                    )
                    save_compressed(self.validation_cache_path, val_dataset)
                logger.info(f"Dataset saved!")
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        # Check if cache exists
        if self.cfg.is_debug:
            # Load and preprocess data
            logger.info(f"Debug mode is enabled. Loading sample data...")
            train_dataset, val_dataset = self._load_and_preprocess_data(
                skip_train=self.skip_train
            )
        else:
            # Load training dataset
            if not self.skip_train:
                assert os.path.exists(
                    self.training_cache_path
                ), f"Cache file does not exist: {self.training_cache_path}"
                logger.info(f"Loading dataset from cache {self.training_cache_path}...")
                train_dataset: Dataset = read_compressed(self.training_cache_path)
                logger.info(
                    f"Loaded training dataset (len:{len(train_dataset)}) from cache {self.training_cache_path}"
                )

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
            self.cfg.dir_path, self.cfg.name, self.cfg.train.q_word_range_file
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
            self.cfg.dir_path, self.cfg.name, self.cfg.train.q_phrase_range_file
        )
        if os.path.exists(file_name):
            q_phrase_ranges = file_utils.read_pickle_file(file_name)
        file_name = os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_phrase_range_file
        )
        if os.path.exists(file_name):
            d_phrase_ranges = file_utils.read_pickle_file(file_name)

        # Shuffle data to avoid qid repetition in the mini-batch
        if self.cfg.is_debug or self.skip_train:
            indices = None
        else:
            indices = self._shuffle_data_to_avoid_qid_repetition(train_dataset)

        # Create DatasetWrapper
        if self.skip_train:
            train_dataset = []
        else:
            train_dataset = DatasetWrapper(
                train_dataset,
                indices=indices,
                q_word_ranges=q_word_ranges,
                q_phrase_ranges=q_phrase_ranges,
                d_word_ranges=d_word_ranges,
                d_phrase_ranges=d_phrase_ranges,
                corpus_mapping=self.corpus_mapping,
                query_mapping=self.query_mapping,
                nway=self.cfg_training.nway,
                data_path=os.path.join(
                    self.cfg.dir_path, self.cfg.name, self.cfg.train.data_file
                ),
                cache_nway=self.cfg.cache_nway,
                q_skip_ids=self.q_tokenizer.special_toks_ids,
                d_skip_ids=self.d_tokenizer.special_toks_ids
                + self.d_tokenizer.punctuations,
                granularity_level=self.cfg_global.model.granularity_level,
                is_use_fine_grained_loss=self.cfg_global.model.is_use_fine_grained_loss,
            )

        val_dataset = DatasetWrapper(
            val_dataset,
            q_word_ranges=q_word_ranges,
            q_phrase_ranges=q_phrase_ranges,
            d_word_ranges=d_word_ranges,
            d_phrase_ranges=d_phrase_ranges,
            corpus_mapping=self.corpus_mapping,
            query_mapping=self.query_mapping,
            nway=self.cfg.val.override_nway,
            data_path=os.path.join(
                self.cfg.dir_path, self.cfg.name, self.cfg.val.data_file
            ),
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg_training.per_device_train_batch_size,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg_training.per_device_eval_batch_size,
            num_workers=2,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg_training.per_device_eval_batch_size,
            num_workers=2,
            collate_fn=collate_fn,
        )
