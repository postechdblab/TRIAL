import abc
import functools
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
    get_indices_to_avoid_repeated_qids_in_minibatch,
    read_compressed,
    read_corpus,
    read_qrels_qids,
    read_queries,
    save_compressed,
)
from eagle.dataset.wrapper import (
    DatasetWrapperForCrossEncoder,
    DatasetWrapperForEAGLE,
    DatasetWrapperForColBERT,
)
from eagle.tokenizer import Tokenizers

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

    @property
    def corpus_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.corpus_file)

    @property
    def queries_path(self) -> str:
        return os.path.join(self.cfg.dir_path, self.cfg.name, self.cfg.query_file)

    @property
    def corpus_cache_path(self) -> str:
        return self.corpus_path + f".{self.tokenizers.model_name}-tok.cache"

    @property
    def debug_corpus_cache_path(self) -> str:
        return self.corpus_cache_path + DEBUG_FILE_SUFFIX

    @property
    def queries_cache_path(self) -> str:
        return self.queries_path + f".{self.tokenizers.model_name}-tok.cache"

    @property
    def debug_queries_cache_path(self) -> str:
        return self.queries_cache_path + DEBUG_FILE_SUFFIX

    @property
    def q_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.q_phrase_range_file
        )

    @property
    def debug_q_phrase_range_path(self) -> str:
        return self.q_phrase_range_path + DEBUG_FILE_SUFFIX

    @property
    def d_phrase_range_path(self) -> str:
        return os.path.join(
            self.cfg.dir_path, self.cfg.name, self.cfg.d_phrase_range_file
        )

    @property
    def debug_d_phrase_range_path(self) -> str:
        return self.d_phrase_range_path + DEBUG_FILE_SUFFIX

    @property
    def dataset_wrapper_cls(self) -> Type[BaseDataset]:
        if self.cfg_global.model.name == "eagle":
            return DatasetWrapperForEAGLE
        elif self.cfg_global.model.name == "cross_encoder":
            return DatasetWrapperForCrossEncoder
        elif self.cfg_global.model.name == "colbert":
            return DatasetWrapperForColBERT
        raise ValueError(f"Invalid model name: {self.cfg_global.model.name}")

    @functools.cached_property
    def corpus_mapping(self) -> Dict[str, int]:
        """Map document keys to indices."""
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
        """Map query keys to indices."""
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
            pids |= set(datum[1:])

        return qids, pids

    def prepare_data(self) -> None:
        """
        Preprocess data for single process before spawning.
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
                tokenizer=self.tokenizers.q_tokenizer, corpus=q_corpus
            )
            # Write the cache
            logger.info(
                f"Saving {len(q_items)} tokenized query cache to {self.queries_cache_path}"
            )
            save_compressed(self.queries_cache_path, q_items)

        # Create cache for debugging
        if self.is_debug:
            # Load qids and pids to be used during debug
            qids_for_debug = pids_for_debug = None
            if any(
                [
                    not os.path.exists(self.debug_corpus_cache_path),
                    not os.path.exists(self.debug_queries_cache_path),
                    not os.path.exists(self.debug_q_phrase_range_path),
                    not os.path.exists(self.debug_d_phrase_range_path),
                ]
            ):
                logger.info(f"Loading qids and pids for debug...")
                qids_for_debug, pids_for_debug = self._load_debug_qids_and_pids()

            # Create debug cache for corpus and queries
            if not os.path.exists(self.debug_corpus_cache_path) or not os.path.exists(
                self.debug_queries_cache_path
            ):
                logger.info(f"Creating debug corpus and queries cache...")

                # Load full cache
                tokenized_corpus = read_compressed(self.corpus_cache_path)
                tokenized_queries = read_compressed(self.queries_cache_path)

                debug_tokenized_corpus = {}
                debug_tokenized_queries = {}

                # Get data from tokenized queries
                key_type = type(list(tokenized_queries.keys())[0])
                for qid in qids_for_debug:
                    if key_type == str:
                        qid = str(qid)
                    debug_tokenized_queries[qid] = tokenized_queries[qid]

                # Get data from tokenized corpus
                key_type = type(list(tokenized_corpus.keys())[0])
                for pid in pids_for_debug:
                    if key_type == str:
                        pid = str(pid)
                    debug_tokenized_corpus[pid] = tokenized_corpus[pid]

                # Save debug cache
                save_compressed(self.debug_corpus_cache_path, debug_tokenized_corpus)
                save_compressed(self.debug_queries_cache_path, debug_tokenized_queries)

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
                q_phrase_ranges_sampled = [
                    q_phrase_ranges[q_key_type(key)] for key in qids_for_debug
                ]
                d_phrase_ranges_sampled = [
                    d_phrase_ranges[d_key_type(key)] for key in pids_for_debug
                ]
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
        # Load tokenized corpus and queries
        logger.info(f"Loading tokenized corpus and queries...")

        if self.is_debug:
            tokenized_corpus_path = self.debug_corpus_cache_path
            tokenized_queries_path = self.debug_queries_cache_path
        else:
            tokenized_corpus_path = self.corpus_cache_path
            tokenized_queries_path = self.queries_cache_path

        # Load corpus and queries
        tokenized_corpus = read_compressed(tokenized_corpus_path)
        tokenized_queries = read_compressed(tokenized_queries_path)

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
        q_phrase_ranges = d_phrase_ranges = None
        if self.cfg_global.model.name == "eagle":
            logger.info(f"Loading phrase ranges...")
            if self.is_debug:
                q_phrase_range_path = self.debug_q_phrase_range_path
                d_phrase_range_path = self.debug_d_phrase_range_path
            else:
                q_phrase_range_path = self.q_phrase_range_path
                d_phrase_range_path = self.d_phrase_range_path

            # Load phrase ranges
            q_phrase_ranges: List[List[Tuple[int, int]]] = file_utils.read_pickle_file(
                q_phrase_range_path
            )
            d_phrase_ranges = file_utils.read_pickle_file(d_phrase_range_path)

        # Shuffle data to avoid qid repetition in the mini-batch
        indices = None
        if not self.is_debug and not self.skip_train:
            logger.info(f"Shuffling data to avoid qid repetition in the mini-batch...")
            indices = self.get_shuffled_indices_to_avoid_qid_repetition(train_dataset)

        # Configs
        add_kwargs = {}
        if self.cfg_global.model.name == "eagle":
            add_kwargs = dict(
                q_phrase_ranges=q_phrase_ranges,
                d_phrase_ranges=d_phrase_ranges,
            )

        # Create DatasetWrapper
        if not self.skip_train:
            train_dataset = self.dataset_wrapper_cls(
                dataset=train_dataset,
                indices=indices,
                nway=self.cfg_global.training.nway,
                cache_nway=self.cfg.cache_nway,
                corpus_mapping=self.corpus_mapping,
                query_mapping=self.query_mapping,
                q_skip_ids=self.tokenizers.q_tokenizer.skip_tok_ids,
                d_skip_ids=self.tokenizers.d_tokenizer.skip_tok_ids,
                model_name=self.cfg_global.model.name,
                **add_kwargs,
            )

        val_dataset = self.dataset_wrapper_cls(
            dataset=val_dataset,
            nway=self.cfg.val.override_nway,
            cache_nway=self.cfg.val.override_nway,
            corpus_mapping=self.corpus_mapping,
            query_mapping=self.query_mapping,
            q_skip_ids=self.tokenizers.q_tokenizer.skip_tok_ids,
            d_skip_ids=self.tokenizers.d_tokenizer.skip_tok_ids,
            model_name=self.cfg_global.model.name,
            **add_kwargs,
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
            num_workers=8,
            collate_fn=self.dataset_wrapper_cls.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=4,
            collate_fn=self.dataset_wrapper_cls.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg_global.training.per_device_eval_batch_size,
            num_workers=1,
            collate_fn=self.dataset_wrapper_cls.collate_fn,
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
