import abc
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.pl_module.utils import tokenize_and_cache_corpus
from eagle.dataset.utils import (
    read_compressed,
    read_corpus,
    read_queries,
    save_compressed,
)
from eagle.model.batch import BaseBatch, BatchForColBERT, BatchForDPR, BatchForEAGLE
from eagle.tokenization import Tokenizers

logger = logging.getLogger("DataModule")

DEBUG_FILE_SUFFIX = ".debug"


class BaseDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, skip_train: bool = False):
        super().__init__()
        self.cfg_global: DictConfig = cfg
        self.cfg: DictConfig = cfg.dataset
        self.tokenizers = Tokenizers(
            cfg.tokenizers.query, cfg.tokenizers.document, cfg.model.backbone_name
        )
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.skip_train = skip_train

    @property
    def is_debug(self) -> bool:
        return self.cfg.is_debug

    # Paths
    @property
    def corpus_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.corpus_file)

    @property
    def queries_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.query_file)

    @property
    def tokenized_corpus_cache_path(self) -> str:
        return self.corpus_path + f".{self.tokenizers.model_name}-tok.cache"

    @property
    def debug_tokenized_corpus_cache_path(self) -> str:
        suffix = ""
        if self.cfg_global.training.use_distillation:
            suffix = ".distillation"
        return self.tokenized_corpus_cache_path + suffix + DEBUG_FILE_SUFFIX

    @property
    def target_tokenized_corpus_cache_path(self) -> str:
        if self.is_debug:
            return self.debug_tokenized_corpus_cache_path
        return self.tokenized_corpus_cache_path

    @property
    def tokenized_queries_cache_path(self) -> str:
        return self.queries_path + f".{self.tokenizers.model_name}-tok.cache"

    @property
    def debug_tokenized_queries_cache_path(self) -> str:
        suffix = ""
        if self.cfg_global.training.use_distillation:
            suffix = ".distillation"
        return self.tokenized_queries_cache_path + suffix + DEBUG_FILE_SUFFIX

    @property
    def target_tokenized_queries_cache_path(self) -> str:
        if self.is_debug:
            return self.debug_tokenized_queries_cache_path
        return self.tokenized_queries_cache_path

    @property
    def q_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_phrase_range_file
        )

    @property
    def debug_q_phrase_range_path(self) -> str:
        return self.q_phrase_range_path + DEBUG_FILE_SUFFIX

    @property
    def target_q_phrase_range_path(self) -> str:
        if self.is_debug:
            return self.debug_q_phrase_range_path
        return self.q_phrase_range_path

    @property
    def d_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_phrase_range_file
        )

    @property
    def debug_d_phrase_range_path(self) -> str:
        return self.d_phrase_range_path + DEBUG_FILE_SUFFIX

    @property
    def target_d_phrase_range_path(self) -> str:
        if self.is_debug:
            return self.debug_d_phrase_range_path
        return self.d_phrase_range_path

    @property
    def data_batching_cls(self) -> Type[BaseBatch]:
        if self.cfg_global.model.name == "eagle":
            return BatchForEAGLE
        elif self.cfg_global.model.name == "dpr":
            return BatchForDPR
        elif self.cfg_global.model.name == "colbert":
            return BatchForColBERT
        raise ValueError(f"Invalid model name: {self.cfg_global.model.name}")

    def _load_debug_qids_and_pids(self) -> Tuple[List, List]:
        # Sample 1000 data
        train_dataset_sampled: BaseDataset = self._load_train_data(
            tokenized_queries=None, tokenized_corpus=None
        )
        val_dataset_sampled: BaseDataset = self._load_val_data(
            tokenized_queries=None, tokenized_corpus=None
        )
        # Filter with debug data
        qids = set()
        pids = set()
        for datum in train_dataset_sampled.data + val_dataset_sampled.data:
            qids.add(datum[0])
            # Extract pids
            if type(datum[1]) == list:
                # Extract pids for distillation data
                list_of_pids = [d[0] for d in datum[1:]]
                pids |= set(list_of_pids)
            else:
                # Extract pids for contrastive data
                pids |= set(datum[1:])

        return qids, pids

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
        """
        # Create a tokenized cache for entire document corpus
        if not os.path.exists(self.tokenized_corpus_cache_path):
            logger.info(f"Reading all documents...")
            d_corpus = read_corpus(self.corpus_path)
            logger.info("Tokenizing and caching document corpus...")
            d_items = tokenize_and_cache_corpus(
                tokenizer=self.tokenizers.d_tokenizer, corpus=d_corpus
            )
            # Write the cache
            logger.info(
                f"Saving {len(d_items)} tokenized document cache to {self.tokenized_corpus_cache_path}"
            )
            save_compressed(self.tokenized_corpus_cache_path, d_items)

        # Create a tokenized cache for entire query set
        if not os.path.exists(self.tokenized_queries_cache_path):
            logger.info("Reading all queries...")
            q_corpus = read_queries(self.queries_path)
            logger.info("Tokenizing and caching query set...")
            q_items = tokenize_and_cache_corpus(
                tokenizer=self.tokenizers.q_tokenizer, corpus=q_corpus
            )
            # Write the cache
            logger.info(
                f"Saving {len(q_items)} tokenized query cache to {self.tokenized_queries_cache_path}"
            )
            save_compressed(self.tokenized_queries_cache_path, q_items)

        # Create toy cache (i.e., small sampled data) for debugging
        if self.is_debug:
            # Load qids and pids to be used during debug
            qids_for_debug = pids_for_debug = None
            if any(
                [
                    not os.path.exists(self.debug_tokenized_corpus_cache_path),
                    not os.path.exists(self.debug_tokenized_queries_cache_path),
                    not os.path.exists(self.debug_q_phrase_range_path),
                    not os.path.exists(self.debug_d_phrase_range_path),
                ]
            ):
                logger.info(f"Loading qids and pids for debug...")
                qids_for_debug, pids_for_debug = self._load_debug_qids_and_pids()

            # Create debug cache for corpus and queries
            if not os.path.exists(
                self.debug_tokenized_corpus_cache_path
            ) or not os.path.exists(self.debug_tokenized_queries_cache_path):
                logger.info(f"Creating debug corpus and queries cache...")

                # Load full cache
                tokenized_corpus = read_compressed(self.tokenized_corpus_cache_path)
                tokenized_queries = read_compressed(self.tokenized_queries_cache_path)

                debug_tokenized_corpus = {}
                debug_tokenized_queries = {}

                # Get data from tokenized queries
                key_type = type(list(tokenized_queries.keys())[0])
                for qid in qids_for_debug:
                    if key_type == str:
                        qid = str(qid)
                    debug_tokenized_queries[str(qid)] = tokenized_queries[qid]

                # Get data from tokenized corpus
                key_type = type(list(tokenized_corpus.keys())[0])
                for pid in pids_for_debug:
                    if key_type == str:
                        pid = str(pid)
                    debug_tokenized_corpus[str(pid)] = tokenized_corpus[pid]

                # Save debug cache
                save_compressed(
                    self.debug_tokenized_corpus_cache_path, debug_tokenized_corpus
                )
                save_compressed(
                    self.debug_tokenized_queries_cache_path, debug_tokenized_queries
                )

            if not os.path.exists(self.debug_q_phrase_range_path) or not os.path.exists(
                self.debug_d_phrase_range_path
            ):
                logger.info(f"Creating debug phrase ranges...")

                # Load the phrase ranges and filter with debug data
                q_phrase_ranges: Dict[str, List[Tuple[int, int]]] = (
                    file_utils.read_pickle_file(self.q_phrase_range_path)
                )
                d_phrase_ranges = file_utils.read_pickle_file(self.d_phrase_range_path)

                # Figure out the key types
                q_key_type = type(list(q_phrase_ranges.keys())[0])
                d_key_type = type(list(d_phrase_ranges.keys())[0])

                # Get the phrase ranges for the sampled data
                q_phrase_ranges_sampled = {
                    key: q_phrase_ranges[q_key_type(key)] for key in qids_for_debug
                }
                d_phrase_ranges_sampled = {
                    key: d_phrase_ranges[d_key_type(key)] for key in pids_for_debug
                }
                del q_phrase_ranges
                del d_phrase_ranges

                # Save the debug phrase ranges
                logger.info(
                    f"Saving {len(q_phrase_ranges_sampled)} and {len(d_phrase_ranges_sampled)} sampled phrase ranges..."
                )
                file_utils.write_pickle_file(
                    q_phrase_ranges_sampled, self.debug_q_phrase_range_path
                )
                file_utils.write_pickle_file(
                    d_phrase_ranges_sampled, self.debug_d_phrase_range_path
                )

        logger.info(f"Dataset preprocessed and saved!")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Preprocess data for each process after spawning.
        Load cache if not debug. Otherwise, load sample data.
        """
        # Load tokenized queries and corpus
        logger.info(f"Loading tokenized queries and corpus...")
        tokenized_corpus = read_compressed(self.target_tokenized_corpus_cache_path)
        tokenized_queries = read_compressed(self.target_tokenized_queries_cache_path)

        # Load phrase ranges for queries and corpus
        phrase_ranges_queries = phrase_ranges_corpus = None
        if self.cfg_global.model.name == "eagle":
            phrase_ranges_queries = file_utils.read_pickle_file(
                self.target_q_phrase_range_path
            )
            phrase_ranges_corpus = file_utils.read_pickle_file(
                self.target_d_phrase_range_path
            )

        # Load train and val data
        train_dataset = None
        if not self.skip_train:
            train_dataset: BaseDataset = self._load_train_data(
                tokenized_queries=tokenized_queries,
                tokenized_corpus=tokenized_corpus,
            )
        val_dataset: BaseDataset = self._load_val_data(
            tokenized_queries=tokenized_queries,
            tokenized_corpus=tokenized_corpus,
        )

        # Shuffle data to avoid qid repetition in the mini-batch
        if not self.is_debug and not self.skip_train:
            logger.info(
                f"Shuffling training data to avoid qid repetition in the mini-batch..."
            )
            train_dataset.shuffle_indices_to_avoid_qid_repetition(
                qrels_path=self.train_qrels_path,
                bsize=self.cfg_global.training.per_device_train_batch_size,
            )

        # Get additional kwargs for the batch
        add_kwargs = {}
        if self.cfg_global.model.name == "eagle":
            add_kwargs["phrase_ranges_queries"] = phrase_ranges_queries
            add_kwargs["phrase_ranges_corpus"] = phrase_ranges_corpus

        # Create Dataset Batch
        train_batch = None
        if not self.skip_train:
            train_batch = self.data_batching_cls(
                dataset=train_dataset,
                skip_tok_ids=self.tokenizers.skip_tok_ids,
                pad_to_max_length=self.cfg_global.training.pad_to_max_length,
                **add_kwargs,
            )

        val_batch = self.data_batching_cls(
            dataset=val_dataset,
            skip_tok_ids=self.tokenizers.skip_tok_ids,
            pad_to_max_length=self.cfg_global.training.pad_to_max_length,
            **add_kwargs,
        )

        # Save datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = val_dataset
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.test_batch = val_batch

    def train_dataloader(self) -> Union[DataLoader, None]:
        if self.skip_train:
            return None
        return DataLoader(
            self.train_batch,
            batch_size=self.cfg_global.training.per_device_train_batch_size,
            num_workers=self.cfg_global.training.train_num_workers,
            collate_fn=self.train_batch.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_batch,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=self.cfg_global.training.val_num_workers,
            collate_fn=self.val_batch.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_batch,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=self.cfg_global.training.test_num_workers,
            collate_fn=self.test_batch.collate_fn,
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
