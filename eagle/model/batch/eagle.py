from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset import BaseDataset
from eagle.model.batch import BaseBatch
from eagle.model.batch.utils import (
    collate_ranges,
    combined_phrase_ranges_into_one_sentence,
    convert_range_to_scatter,
    get_mask,
)


class BatchForEAGLE(BaseBatch):
    def __init__(
        self,
        dataset: BaseDataset,
        skip_tok_ids: List[int],
        phrase_ranges_queries,
        phrase_ranges_corpus,
    ):
        super().__init__(dataset=dataset, skip_tok_ids=skip_tok_ids)
        self.phrase_ranges_queries = phrase_ranges_queries
        self.phrase_ranges_corpus = phrase_ranges_corpus

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

    def parse_data(self, data: List[Any]) -> Dict[str, Any]:
        labels = data.get("labels", None)
        distillation_scores = data.get("distillation_scores", None)

        qid = data["q_id"]
        q_tok_ids = data["q_tok_ids"]
        doc_tok_ids = data["doc_tok_ids"]
        pos_doc_ids = data["pos_doc_ids"]
        neg_doc_ids = data["neg_doc_ids"]
        q_sent_start_indices = data["q_sent_start_indices"]
        doc_sent_start_indices = data["doc_sent_start_indices"]

        # Get token masks
        q_tok_mask = get_mask(input_ids=q_tok_ids, skip_ids=self.skip_tok_ids)
        q_tok_att_mask = get_mask(input_ids=q_tok_ids, skip_ids=[0])
        doc_tok_mask = get_mask(input_ids=doc_tok_ids, skip_ids=self.skip_tok_ids)
        doc_tok_att_mask = get_mask(input_ids=doc_tok_ids, skip_ids=[0])

        # Get phrase ranges
        q_phrase_ranges: List[Tuple] = combined_phrase_ranges_into_one_sentence(
            self.phrase_ranges_queries[self.phrase_ranges_queries_key_type(qid)]
        )
        doc_phrase_ranges: List[List[Tuple]] = [
            combined_phrase_ranges_into_one_sentence(
                self.phrase_ranges_corpus[self.phrase_ranges_corpus_key_type(pid)]
            )
            for pid in pos_doc_ids + neg_doc_ids
        ]

        # Get phrase masks
        q_phrase_mask = torch.ones(len(q_phrase_ranges), dtype=torch.bool).float()
        doc_phrase_mask = [
            torch.ones(len(dpr), dtype=torch.bool).float() for dpr in doc_phrase_ranges
        ]
        doc_phrase_mask = pad_sequence(doc_phrase_mask, batch_first=True)

        # Get sentence masks
        q_sent_mask = torch.ones(len(q_sent_start_indices), dtype=torch.bool).float()
        doc_sent_mask = [
            torch.ones(len(dssi), dtype=torch.bool).float()
            for dssi in doc_sent_start_indices
        ]

        # Get scatter indices for phrases
        q_phrase_scatter_indices: List[int] = convert_range_to_scatter(q_phrase_ranges)
        doc_phrase_scatter_indices: List[List[int]] = [
            convert_range_to_scatter(item) for item in doc_phrase_ranges
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
        }

    @staticmethod
    def _collate_q_tok_ids(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_q_tok_att_mask(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_q_tok_mask(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_q_phrase_mask(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_q_sent_mask(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_q_phrase_scatter_indices(data: List[List[int]]) -> List[torch.Tensor]:
        padded_values = collate_ranges(
            [torch.tensor(item, dtype=torch.int32, device="cpu") for item in data]
        )
        return padded_values

    @staticmethod
    def _collate_q_sent_start_indices(data: List[List[int]]) -> List[List[int]]:
        return data

    @staticmethod
    def _collate_doc_tok_ids(data: List[torch.Tensor]) -> torch.Tensor:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    @staticmethod
    def _collate_doc_tok_att_mask(data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    @staticmethod
    def _collate_doc_tok_mask(data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    @staticmethod
    def _collate_doc_phrase_mask(data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    @staticmethod
    def _collate_doc_sent_mask(data: List[torch.Tensor]) -> List[torch.Tensor]:
        """Data shape: [bsize, num_docs, num_toks]"""
        bsize, num_docs = len(data), len(data[0])
        # Convert to list of list of tensors
        flattened_data = list_utils.do_flatten_list([item for item in data])
        # Pad the sequence to the maximum length
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    @staticmethod
    def _collate_doc_phrase_scatter_indices(data: List[Any]) -> List[torch.Tensor]:
        padded_values = list_utils.do_flatten_list([item for item in data])
        padded_values = collate_ranges(
            [
                torch.tensor(item, dtype=torch.int32, device="cpu")
                for item in padded_values
            ]
        )
        return padded_values

    @staticmethod
    def _collate_doc_sent_start_indices(
        data: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        return data

    @staticmethod
    def _collate_labels(data: List[torch.Tensor]) -> List[torch.Tensor]:
        return pad_sequence(data, batch_first=True)

    @staticmethod
    def _collate_distillation_scores(data: List[Any]) -> List[Any]:
        return data

    @staticmethod
    def collate_fn(input_dics: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            new_dict[key] = collate_method([dic[key] for dic in input_dics])
        return new_dict
