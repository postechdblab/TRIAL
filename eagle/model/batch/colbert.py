from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset import BaseDataset
from eagle.model.batch import BaseBatch
from eagle.model.batch.utils import (
    add_doc_ranges_and_mask,
    add_query_ranges_and_mask,
    collate_ranges,
    get_mask,
)


class BatchForColBERT(BaseBatch):
    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        labels = data.get("labels", None)
        distillation_scores = data.get("distillation_scores", None)

        qid = data["q_id"]
        q_tok_ids = data["q_tok_ids"]
        doc_tok_ids = data["doc_tok_ids"]
        pos_doc_ids = data["pos_doc_ids"]
        neg_doc_ids = data["neg_doc_ids"]

        # Get token masks
        q_tok_mask = get_mask(input_ids=q_tok_ids, skip_ids=self.skip_tok_ids)
        q_tok_att_mask = get_mask(input_ids=q_tok_ids, skip_ids=[0])
        doc_tok_mask = get_mask(input_ids=doc_tok_ids, skip_ids=self.skip_tok_ids)
        doc_tok_att_mask = get_mask(input_ids=doc_tok_ids, skip_ids=[0])

        return {
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "q_tok_mask": q_tok_mask,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "doc_tok_mask": doc_tok_mask,
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
    def _collate_q_sent_mask(data: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(data, batch_first=True)

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
            if not hasattr(BatchForColBERT, collate_method_name):
                raise ValueError(f"Unsupported key: {key}")
            # Perform the collation
            collate_method = getattr(BatchForColBERT, collate_method_name)
            new_dict[key] = collate_method([dic[key] for dic in input_dics])
        return new_dict
