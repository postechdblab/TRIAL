from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.utils import get_att_mask, get_mask
from eagle.model.batch import BaseBatch


class BatchForColBERT(BaseBatch):
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
        return pad_sequence(data, batch_first=True)

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
        padded_data = pad_sequence(flattened_data, batch_first=True)
        # Convert to the original shape
        return padded_data.reshape(bsize, num_docs, -1)

    def _collate_labels(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        return torch.stack(data)

    def _collate_distillation_scores(self, data: List[Any]) -> List[Any]:
        return torch.stack(data)

    def _collate_pos_doc_ids(self, data: List[List[str]]) -> List[List[str]]:
        return data

    def collate_fn(
        self,
        input_dics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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
            new_dict[key] = collate_method(self, [dic[key] for dic in input_dics])
        return new_dict

    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        labels = data.get("labels", None)
        distillation_scores = data.get("distillation_scores", None)

        qid = data["q_id"]
        q_tok_ids = data["q_tok_ids"]
        pos_doc_ids = data["pos_doc_ids"]
        doc_tok_ids = data.get("doc_tok_ids", None)
        neg_doc_ids = data.get("neg_doc_ids", None)

        # Pad the input ids to the maximum length
        if self.pad_to_max_length:
            q_tok_ids = self.dataset.tokenizers.q_tokenizer.pad_sequence_by_max_len(
                q_tok_ids
            )
            if doc_tok_ids is not None:
                doc_tok_ids = (
                    self.dataset.tokenizers.d_tokenizer.pad_sequence_by_max_len(
                        doc_tok_ids
                    )
                )

        # Get token masks
        q_tok_mask = get_mask(input_ids=q_tok_ids, skip_ids=self.skip_tok_ids)
        q_tok_att_mask = get_att_mask(input_ids=q_tok_ids, skip_ids=[0])
        doc_tok_mask = doc_tok_att_mask = None
        if doc_tok_ids is not None:
            doc_tok_mask = get_mask(input_ids=doc_tok_ids, skip_ids=self.skip_tok_ids)
            doc_tok_att_mask = get_att_mask(input_ids=doc_tok_ids, skip_ids=[0])

        return {
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "q_tok_mask": q_tok_mask,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "doc_tok_mask": doc_tok_mask,
            "labels": labels,
            "distillation_scores": distillation_scores,
            "pos_doc_ids": pos_doc_ids,
        }
