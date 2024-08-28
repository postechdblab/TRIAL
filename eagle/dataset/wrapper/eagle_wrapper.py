import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.utils import (add_doc_ranges_and_mask,
                                 add_query_ranges_and_mask, collate_ranges)
from eagle.dataset.wrapper import BaseDatasetWrapper

logger = logging.getLogger("DatasetWrapper")


class DatasetWrapperForEAGLE(BaseDatasetWrapper):
    def __init__(
        self,
        dataset: BaseDataset,
        indices: Optional[List[int]] = None,
        q_word_ranges: Optional[List[Tuple[int, int]]] = None,
        q_phrase_ranges: Optional[List[Tuple[int, int]]] = None,
        d_word_ranges: Optional[List[Tuple[int, int]]] = None,
        d_phrase_ranges: Optional[List[Tuple[int, int]]] = None,
        corpus_mapping: Dict[int, int] = None,
        query_mapping: Dict[int, int] = None,
        nway: int = None,
        cache_nway: int = None,
        q_skip_ids: List[int] = None,
        d_skip_ids: List[int] = None,
        model_name: str = None,
    ):
        super(DatasetWrapperForEAGLE, self).__init__(dataset=dataset, 
                                                     indices=indices, 
                                                     nway=nway, 
                                                     cache_nway=cache_nway, 
                                                     q_skip_ids=q_skip_ids, 
                                                     d_skip_ids=d_skip_ids,
                                                     query_mapping=query_mapping,
                                                     corpus_mapping=corpus_mapping)
        self.q_word_ranges = q_word_ranges
        self.q_phrase_ranges = q_phrase_ranges
        self.d_word_ranges = d_word_ranges
        self.d_phrase_ranges = d_phrase_ranges
        self.query_mapping = query_mapping
        self.corpus_mapping = corpus_mapping
        self.model_name = model_name
        # Check if variables are valid
        assert (
            "neg_doc_ids" not in self.dataset[0] 
            or len(self.dataset[0]["neg_doc_ids"]) == 0
            or len(self.dataset[0]["neg_doc_ids"]) >= nway - 1
        ), f"nway={nway} is larger than the total doc num: {len(self.dataset[0]["neg_doc_ids"])}"
        assert self.model_name == "eagle", f"model_name={model_name} is not supported"
        if q_word_ranges is not None:
            assert len(self.q_word_ranges) == len(
                self.query_mapping
            ), f"len(self.q_word_ranges)={len(self.q_word_ranges)}, len(self.dataset)={len(self.dataset)}"
        if self.q_phrase_ranges is not None:
            assert len(self.q_phrase_ranges) == len(
                self.query_mapping
            ), f"len(self.query_mapping)={len(self.query_mapping)}, len(self.q_phrase_ranges)={len(self.q_phrase_ranges)}"
        if d_word_ranges is not None:
            assert len(self.d_word_ranges) == len(
                self.corpus_mapping
            ), f"len(self.d_word_ranges)={len(self.d_word_ranges)}, len(self.dataset)={len(self.dataset)}"
        if d_phrase_ranges is not None:
            assert len(self.d_phrase_ranges) == len(
                self.corpus_mapping
            ), f"len(self.corpus_mapping)={len(self.corpus_mapping)}, len(self.d_phrase_ranges)={len(self.d_phrase_ranges)}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Replace the nway
        if idx > len(self):
            raise IndexError(f"Index {idx} out of range {len(self)}")
        shuff_idx = self.indices[idx]
        data = self.dataset[shuff_idx]
        # Replace the nway
        if self.nway < self.cache_nway:
            data["doc_tok_ids"] = data["doc_tok_ids"][: self.nway]
            data["doc_tok_att_mask"] = data["doc_tok_att_mask"][: self.nway]
            if "distillation_scores" in data:
                data["distillation_scores"] = data["distillation_scores"][: self.nway]

        # Extract meta data
        qid = data["q_id"]
        qidx = self.query_mapping[qid]
        pids = data["pos_doc_ids"]
        if "neg_doc_ids" in data:
            pids.extend(data["neg_doc_ids"])
        pindices = [self.corpus_mapping[pid] for pid in pids]
        pindices = pindices[: self.nway]

        # Extract ranges
        q_word_ranges: List[Tuple] = self.q_word_ranges[qidx]
        q_phrase_ranges: List[Tuple] = self.q_phrase_ranges[qidx]
        d_word_ranges: List[List[Tuple]] = [self.d_word_ranges[i] for i in pindices]
        d_phrase_ranges: List[List[Tuple]] = [self.d_phrase_ranges[i] for i in pindices]

        # Add ranges and token mask
        data = add_query_ranges_and_mask(
            input_dict=data,
            word_ranges=q_word_ranges,
            phrase_ranges=q_phrase_ranges,
            skip_ids=self.q_skip_ids,
            use_coarse_emb=True,
        )
        if "doc_tok_ids" in data:
            data = add_doc_ranges_and_mask(
                input_dict=data,
                word_ranges=d_word_ranges,
                phrase_ranges=d_phrase_ranges,
                skip_ids=self.d_skip_ids,
                use_coarse_emb=True,
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
            if key == "q_scatter_indices":
                padded_values = collate_ranges(
                    [
                        torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                        for dic in input_dics
                    ]
                )
            elif key == "doc_scatter_indices":
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
                padded_values = pad_sequence(values, batch_first=True).unsqueeze(-1) == 0
            elif key in ["q_id", "pos_doc_idxs"]:
                padded_values = [dic[key] for dic in input_dics]
            elif key in ["pos_doc_ids", "neg_doc_ids"]:
                continue
            elif key == "distillation_scores":
                padded_values = torch.nn.functional.log_softmax(
                    torch.tensor(
                        [dic[key] for dic in input_dics], dtype=get_dtype(key), device="cpu"
                    ),
                    dim=-1,
                )
            else:
                raise ValueError(f"Unsupported key: {key}")
            new_dict[key] = padded_values

        return new_dict
