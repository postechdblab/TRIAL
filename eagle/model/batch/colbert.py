from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.model.batch import BaseBatch
from eagle.model.batch.utils import (
    add_doc_ranges_and_mask,
    add_query_ranges_and_mask,
    collate_ranges,
)


class BatchForColBERT(BaseBatch):
    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract meta data
        qid = data["q_id"]
        qidx = self.query_mapping[qid]
        pids = data["pos_doc_ids"]
        if "neg_doc_ids" in data:
            pids.extend(data["neg_doc_ids"])
        pindices = [self.corpus_mapping[pid] for pid in pids]
        pindices = pindices[: self.dataset.nway]

        # Add ranges and token mask
        data = add_query_ranges_and_mask(
            input_dict=data,
            word_ranges=None,
            phrase_ranges=None,
            skip_ids=self.q_skip_ids,
            use_coarse_emb=False,
        )
        if "doc_tok_ids" in data:
            data = add_doc_ranges_and_mask(
                input_dict=data,
                word_ranges=None,
                phrase_ranges=None,
                skip_ids=self.d_skip_ids,
                use_coarse_emb=False,
            )

        # Add index for positive document
        data["pos_doc_idxs"] = [self.corpus_mapping[i] for i in data["pos_doc_ids"]]

        return data

    @staticmethod
    def collate_fn(input_dics: List[Dict]) -> Dict:
        """Collate list of dictionaries into a single dictionary."""

        def get_dtype(key: str) -> torch.dtype:
            if "mask" in key or key == "labels":
                return torch.bool
            elif "id" in key:
                return torch.int32
            elif "scores" in key:
                return torch.float32
            return torch.long

        new_dict = {}
        # Assume all dictionaries have the same keys
        keys = list(input_dics[0].keys())
        # Collate for each key
        for key in keys:
            if input_dics[0][key] is None:
                new_dict[key] = None
                continue
            if key == "q_phrase_scatter_indices":
                padded_values = collate_ranges(
                    [
                        torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                        for dic in input_dics
                    ]
                )
            elif key == "doc_phrase_scatter_indices":
                padded_values = list_utils.do_flatten_list(
                    [input_dic[key] for input_dic in input_dics]
                )
                padded_values = collate_ranges(
                    [
                        torch.tensor(item, dtype=get_dtype(key), device="cpu")
                        for item in padded_values
                    ]
                )
            elif key in ["q_tok_ids", "q_tok_att_mask", "labels"]:
                values = [
                    torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                    for dic in input_dics
                ]
                padded_values = pad_sequence(values, batch_first=True)
            elif key in ["doc_tok_ids", "doc_tok_att_mask", "tok_ids", "tok_att_mask"]:
                values = []
                for input_dic in input_dics:
                    for item in input_dic[key]:
                        values.append(
                            torch.tensor(item, dtype=get_dtype(key), device="cpu")
                        )
                padded_values = pad_sequence(values, batch_first=True)
                padded_values = padded_values.reshape(
                    len(input_dics), -1, padded_values.shape[1]
                )
            elif key in ["q_tok_mask", "q_phrase_mask"]:
                values = [dic[key].clone().detach().unsqueeze(-1) for dic in input_dics]
                padded_values = pad_sequence(values, batch_first=True) == 0
            elif key in ["doc_tok_mask", "doc_phrase_mask", "tok_mask"]:
                values = list_utils.do_flatten_list(
                    [torch.unbind(dic[key].clone().detach()) for dic in input_dics]
                )
                padded_values = (
                    pad_sequence(values, batch_first=True).unsqueeze(-1) == 0
                )
            elif key in ["q_id", "pos_doc_idxs"]:
                padded_values = [dic[key] for dic in input_dics]
            elif key in ["pos_doc_ids", "neg_doc_ids"]:
                continue
            elif key == "distillation_scores":
                padded_values = torch.nn.functional.log_softmax(
                    torch.tensor(
                        [dic[key] for dic in input_dics],
                        dtype=get_dtype(key),
                        device="cpu",
                    ),
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported key: {key}")
            new_dict[key] = padded_values

        return new_dict
