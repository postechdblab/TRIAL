import copy
import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.utils import (
    extract_pids_from_msmarco_data,
    extract_pids_from_non_msmarco_data,
    extract_qids_from_msmarco_data,
    extract_qids_from_non_msmarco_data,
    get_att_mask,
    get_mask,
)
from eagle.model.batch import BaseBatch
from eagle.model.batch.utils import (
    collate_ranges,
    convert_range_to_scatter,
    cut_off_phrase_ranges_by_max_len,
)
from eagle.phrase.utils import (
    combined_phrase_ranges_into_one_sentence,
    fill_in_missing_phrase_ranges,
    fix_bad_index_ranges,
)

logger = logging.getLogger("BatchForEAGLE")


class BatchForEAGLE(BaseBatch):
    def __init__(
        self,
        dataset: BaseDataset,
        skip_tok_ids: List[int],
        pad_to_max_length: bool = False,
        phrase_ranges_queries=None,
        phrase_ranges_corpus=None,
    ):
        assert not pad_to_max_length, "Padding to max length is not supported yet."
        super().__init__(
            dataset=dataset,
            skip_tok_ids=skip_tok_ids,
            pad_to_max_length=pad_to_max_length,
        )
        self.phrase_ranges_queries: List[List[Tuple[int, int]]] = phrase_ranges_queries
        self.phrase_ranges_corpus: List[List[Tuple[int, int]]] = phrase_ranges_corpus
        self._remove_redundant_phrase_ranges_queries()
        self._remove_redundant_phrase_ranges_corpus()

    @property
    def phrase_ranges_queries_key_type(self) -> type:
        if len(self.phrase_ranges_queries) == 0:
            return None
        if not hasattr(self, "_phrase_ranges_queries_key_type"):
            self._phrase_ranges_queries_key_type = type(
                list(self.phrase_ranges_queries.keys())[0]
            )
        return self._phrase_ranges_queries_key_type

    @property
    def phrase_ranges_corpus_key_type(self) -> type:
        if len(self.phrase_ranges_corpus) == 0:
            return None
        if not hasattr(self, "_phrase_ranges_corpus_key_type"):
            self._phrase_ranges_corpus_key_type = type(
                list(self.phrase_ranges_corpus.keys())[0]
            )
        return self._phrase_ranges_corpus_key_type

    def _remove_redundant_phrase_ranges_queries(self) -> None:
        """Delete the redundant phrase ranges in the queries for memory saving."""
        # Get qids from the data
        if self.dataset.cfg_dataset.name == "beir-msmarco":
            required_qids: Set[str] = [
                str(item) for item in extract_qids_from_msmarco_data(self.dataset.data)
            ]
        else:
            required_qids: Set[str] = extract_qids_from_non_msmarco_data(
                self.dataset.data
            )
        all_qids: List[Union[int, str]] = list(self.phrase_ranges_queries.keys())
        # Remove redundant phrase ranges in the queries
        new_data: Dict[str, List[Tuple[int, int]]] = {}
        for qid in all_qids:
            if str(qid) in required_qids:
                new_data[qid] = copy.deepcopy(self.phrase_ranges_queries[qid])
        removed_cnt = len(self.phrase_ranges_queries) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for phrase ranges in the queries."
        )
        assert len(new_data) > 0, "Every phrase ranges are filtered out!"
        # Update the data
        self.phrase_ranges_queries = new_data
        return None

    def _remove_redundant_phrase_ranges_corpus(self) -> None:
        """Delete the redundant phrase ranges in the corpus for memory saving."""
        # Get doc ids from the data
        if self.dataset.cfg_dataset.name == "beir-msmarco":
            doc_ids: Set[int] = extract_pids_from_msmarco_data(self.dataset.data)
        else:
            doc_ids: Set[int] = extract_pids_from_non_msmarco_data(self.dataset.data)
        all_pids: List[int] = list(self.phrase_ranges_corpus.keys())
        # Remove redundant phrase ranges in the corpus
        new_data: Dict[int, List[Tuple[int, int]]] = {}
        for pid in all_pids:
            if pid in doc_ids:
                new_data[pid] = copy.deepcopy(self.phrase_ranges_corpus[pid])
        removed_cnt = len(self.phrase_ranges_corpus) - len(new_data)
        logger.info(
            f"Removed {removed_cnt} and {len(new_data)} left for phrase ranges in the corpus."
        )
        assert len(new_data) > 0, "Every phrase ranges are filtered out!"
        # Update the data
        self.phrase_ranges_corpus = new_data
        return None

    def parse_data(self, data: List[Any]) -> Dict[str, Any]:
        """Add phrase indices and masks to the data."""
        # Get data
        labels = data.get("labels", None)
        distillation_scores = data.get("distillation_scores", None)

        qid = data["q_id"]
        q_tok_ids = data["q_tok_ids"]
        pos_doc_ids = data["pos_doc_ids"]
        doc_tok_ids = data.get("doc_tok_ids", None)
        neg_doc_ids = data.get("neg_doc_ids", None)
        q_sent_start_indices = data["q_sent_start_indices"]
        doc_sent_start_indices = data.get("doc_sent_start_indices", None)

        # Dynamic config
        is_to_encode_doc = doc_tok_ids is not None

        # Pad the input ids to the maximum length
        if self.pad_to_max_length:
            q_tok_ids = self.dataset.tokenizers.q_tokenizer.pad_sequence_by_max_len(
                q_tok_ids
            )
            doc_tok_ids = self.dataset.tokenizers.d_tokenizer.pad_sequence_by_max_len(
                doc_tok_ids
            )

        # Get phrase ranges
        q_phrase_ranges: List[Tuple] = combined_phrase_ranges_into_one_sentence(
            [
                fix_bad_index_ranges(item)
                for item in self.phrase_ranges_queries[
                    self.phrase_ranges_queries_key_type(qid)
                ]
            ]
        )
        doc_phrase_ranges = None
        if is_to_encode_doc:
            doc_phrase_ranges: List[List[Tuple]] = [
                combined_phrase_ranges_into_one_sentence(
                    [
                        fix_bad_index_ranges(item)
                        for item in self.phrase_ranges_corpus[
                            self.phrase_ranges_corpus_key_type(pid)
                        ]
                    ]
                )
                for pid in pos_doc_ids + neg_doc_ids
            ]

        # Cut off phrase ranges if it exceeds the maximum length
        q_phrase_ranges = cut_off_phrase_ranges_by_max_len(
            q_phrase_ranges, self.dataset.tokenizers.q_tokenizer.cfg.max_len
        )
        if is_to_encode_doc:
            doc_phrase_ranges = [
                cut_off_phrase_ranges_by_max_len(
                    item, self.dataset.tokenizers.d_tokenizer.cfg.max_len
                )
                for item in doc_phrase_ranges
            ]

        # Fix the missing phrase ranges.
        # This is a temporary fix. Need to change the phrase range creation logic to avoid this.
        q_phrase_ranges = fill_in_missing_phrase_ranges(q_phrase_ranges)
        if is_to_encode_doc:
            doc_phrase_ranges = [
                fill_in_missing_phrase_ranges(item) for item in doc_phrase_ranges
            ]

        # Get scatter indices for phrases
        q_phrase_scatter_indices: List[int] = convert_range_to_scatter(q_phrase_ranges)
        doc_phrase_scatter_indices = None
        if is_to_encode_doc:
            doc_phrase_scatter_indices: List[List[int]] = [
                convert_range_to_scatter(item) for item in doc_phrase_ranges
            ]

        # # Cut off phrase scatter indices if it exceeds the maximum length
        # # TODO: Isn't this redundant as we already cut off by the max length in the phrase range creation?
        # q_phrase_scatter_indices = (
        #     self.dataset.tokenizers.q_tokenizer.cutoff_by_max_len(
        #         q_phrase_scatter_indices,
        #         maintain_special_tokens=False,
        #     )
        # )
        # doc_phrase_scatter_indices = [
        #     self.dataset.tokenizers.d_tokenizer.cutoff_by_max_len(
        #         item,
        #         maintain_special_tokens=False,
        #     )
        #     for item in doc_phrase_scatter_indices
        # ]

        # Create mask
        # Create token masks
        q_tok_mask = get_mask(input_ids=q_tok_ids, skip_ids=self.skip_tok_ids)
        q_tok_att_mask = get_att_mask(input_ids=q_tok_ids, skip_ids=[0])
        doc_tok_mask = doc_tok_att_mask = None
        if is_to_encode_doc:
            doc_tok_mask = get_mask(input_ids=doc_tok_ids, skip_ids=self.skip_tok_ids)
            doc_tok_att_mask = get_att_mask(input_ids=doc_tok_ids, skip_ids=[0])

        # Create phrase masks
        q_phrase_mask = torch.zeros(len(q_phrase_ranges), dtype=torch.bool).float()
        doc_phrase_mask = None
        if is_to_encode_doc:
            doc_phrase_mask = [
                torch.zeros(len(dpr), dtype=torch.bool).float()
                for dpr in doc_phrase_ranges
            ]
            doc_phrase_mask = pad_sequence(
                doc_phrase_mask, batch_first=True, padding_value=1.0
            )

        # Create sentence masks
        doc_sent_mask = None
        q_sent_mask = torch.zeros(len(q_sent_start_indices), dtype=torch.bool).float()
        if is_to_encode_doc:
            doc_sent_mask = [
                torch.zeros(len(dssi), dtype=torch.bool).float()
                for dssi in doc_sent_start_indices
            ]

        return {
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "q_tok_mask": q_tok_mask,
            "q_phrase_mask": q_phrase_mask,
            "q_sent_mask": q_sent_mask,
            "q_phrase_scatter_indices": q_phrase_scatter_indices,
            "q_sent_start_indices": q_sent_start_indices,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "doc_tok_mask": doc_tok_mask,
            "doc_phrase_mask": doc_phrase_mask,
            "doc_sent_mask": doc_sent_mask,
            "doc_phrase_scatter_indices": doc_phrase_scatter_indices,
            "doc_sent_start_indices": doc_sent_start_indices,
            "labels": labels,
            "distillation_scores": distillation_scores,
            "pos_doc_ids": pos_doc_ids,
        }

    def _collate_q_tok_ids(self, data: List[torch.Tensor]) -> torch.Tensor:
        if self.pad_to_max_length:
            return torch.stack(data)
        return pad_sequence(data, batch_first=True)

    def _collate_q_tok_att_mask(self, data: List[torch.Tensor]) -> torch.Tensor:
        if self.pad_to_max_length:
            return torch.stack(data)
        return pad_sequence(data, batch_first=True)

    def _collate_q_tok_mask(self, data: List[torch.Tensor]) -> torch.Tensor:
        if self.pad_to_max_length:
            return torch.stack(data)
        return pad_sequence(data, batch_first=True, padding_value=1.0)

    def _collate_q_phrase_mask(self, data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True, padding_value=1.0)

    def _collate_q_sent_mask(self, data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True, padding_value=1.0)

    def _collate_q_phrase_scatter_indices(
        self, data: List[List[int]]
    ) -> List[torch.Tensor]:
        padded_values = collate_ranges(
            [torch.tensor(item, dtype=torch.long, device="cpu") for item in data]
        )
        return padded_values

    def _collate_q_sent_start_indices(self, data: List[List[int]]) -> List[List[int]]:
        return data

    def _collate_doc_tok_ids(self, data: List[torch.Tensor]) -> torch.Tensor:
        """Data shape: [bsize, num_docs, num_toks]"""
        if self.pad_to_max_length:
            return torch.stack(data)
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_doc_tok_att_mask(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        if self.pad_to_max_length:
            return torch.stack(data)
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_doc_tok_mask(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        if self.pad_to_max_length:
            return torch.stack(data)
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True, padding_value=1.0)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_doc_phrase_mask(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True, padding_value=1.0)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_doc_sent_mask(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True, padding_value=1.0)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_doc_phrase_scatter_indices(
        self, data: List[Any]
    ) -> List[torch.Tensor]:
        padded_values = list_utils.do_flatten_list([item for item in data])
        padded_values = collate_ranges(
            [
                torch.tensor(item, dtype=torch.long, device="cpu")
                for item in padded_values
            ]
        )
        return padded_values

    def _collate_doc_sent_start_indices(
        self,
        data: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        return data

    def _collate_labels(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        return pad_sequence(data, batch_first=True)

    def _collate_distillation_scores(self, data: List[Any]) -> List[Any]:
        return torch.stack(data)

    def _collate_pos_doc_ids(self, data: List[List[str]]) -> List[List[str]]:
        return data

    def collate_fn(self, input_dics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate list of dictionaries into a single dictionary."""
        new_dict = {}
        # Assume all dictionaries have the same keys
        keys = list(input_dics[0].keys())
        # Collate for each key
        for key in keys:
            if input_dics[0][key] is None:
                new_dict[key] = None
                continue
            # Get the correct method to collate
            collate_method_name = f"_collate_{key}"
            # Check if the method exists
            if not hasattr(BatchForEAGLE, collate_method_name):
                raise ValueError(f"Unsupported key: {key}")
            # Perform the collation
            collate_method = getattr(BatchForEAGLE, collate_method_name)
            new_dict[key] = collate_method(self, [dic[key] for dic in input_dics])
        return new_dict
