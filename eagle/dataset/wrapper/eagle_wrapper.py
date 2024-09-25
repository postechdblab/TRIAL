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
        q_phrase_ranges: Optional[List[Tuple[int, int]]] = None,
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
        self.q_phrase_ranges = q_phrase_ranges
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
        if self.q_phrase_ranges is not None:
            assert len(self.q_phrase_ranges) == len(
                self.query_mapping
            ), f"len(self.query_mapping)={len(self.query_mapping)}, len(self.q_phrase_ranges)={len(self.q_phrase_ranges)}"
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
        q_phrase_ranges: List[Tuple] = self.q_phrase_ranges[qidx]
        d_phrase_ranges: List[List[Tuple]] = [self.d_phrase_ranges[i] for i in pindices]

        # Modify the p ranges
        q_phrase_ranges: List[Tuple] = combined_phrase_ranges_into_one_sentence(q_phrase_ranges)
        d_phrase_ranges: List[List[Tuple]] = [combined_phrase_ranges_into_one_sentence(d_phrases) for d_phrases in d_phrase_ranges]

        # Combine split sentences into one
        data = self.modify_data_to_combine_splitted_sentences(data)

        # Add ranges and token mask
        data = add_query_ranges_and_mask(
            input_dict=data,
            phrase_ranges=q_phrase_ranges,
            skip_ids=self.q_skip_ids,
            use_coarse_emb=True,
        )
        if "doc_tok_ids" in data:
            data = add_doc_ranges_and_mask(
                input_dict=data,
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

    def modify_data_to_combine_splitted_sentences(self, data: Dict) -> Dict:
        # Check if the data is already combined. We are getting the shallow copy
        if "is_combined" in data.keys():
            return data

        # Combine sentences
        q_tok_ids: List[int] = combine_splitted_tok_ids(data["q_tok_ids"])

        ## doc_tok_ids
        d_tok_ids:List[List[int]] = [combine_splitted_tok_ids(d) for d in data["doc_tok_ids"]]

        # Replace the data
        data["q_tok_ids"] = q_tok_ids
        data["d_tok_ids"] = d_tok_ids

        # Modify att_mask
        # TODO: Need to check if this is correct
        # raise NotImplementedError("Need to fix the below code.")

        # Create ones fill with q_tok_ids
        data["q_tok_att_mask"] = torch.ones(q_tok_ids)
        data["doc_tok_att_mask"] = torch.ones(d_tok_ids)

        # Create attention mask

        return data

def combined_phrase_ranges_into_one_sentence(phrase_ranges_list: List[List[Tuple[int, int]]]) -> None:
    # Figure out the number to add for each sentences
    number_to_adds = [] 
    for sent_idx, p_ranges in enumerate(phrase_ranges_list):    
        if sent_idx == 0 :
            number_to_adds.append(0)
            continue
        # Get the last number to have the cumulative number
        base_number = number_to_adds[-1]
        # Add the max value of the previous sentence
        previous_token_cnt = phrase_ranges_list[sent_idx-1][-1][[-1]]
        # Remove the first two special tokens if the previous sentence is not the first sentence
        if sent_idx > 1:
            previous_token_cnt - 2
        number_to_adds.append(base_number + previous_token_cnt)
    
    # Modify the token ranges
    for sent_idx, (number_to_add, p_ranges) in enumerate(zip(number_to_adds, phrase_ranges_list)):
        # Create new p_ranges

        # Remove the ranges for the first two tokens
        if sent_idx != 0:
            p_ranges = p_ranges[2:]
        # Modify the token idx
        p_ranges = [(s+number_to_add, e+number_to_add) for s, e in p_ranges]
        
        # update the data
        phrase_ranges_list[sent_idx] =  p_ranges
    
    return phrase_ranges_list

def combine_splitted_tok_ids(tok_ids_list: List[List[int]]) -> List[int]:
    # This needs to be changed when the tokenizer changes
    BEGIN_SPECIAL_TOK_NUM = 2
    # Remove special tokens in front
    tok_ids = [tok_ids[BEGIN_SPECIAL_TOK_NUM:] for tok_ids in tok_ids_list]
    # Combine sentences
    tok_ids = list_utils.do_flatten_list(tok_ids)
    # Append special tokens at the beginning of the sentence only
    tok_ids = tok_ids_list[0][:-BEGIN_SPECIAL_TOK_NUM] + tok_ids
    
    return tok_ids
    
