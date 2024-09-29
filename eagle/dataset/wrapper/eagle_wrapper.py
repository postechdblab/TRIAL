import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.utils import (add_doc_ranges_and_mask,
                                 add_query_ranges_and_mask, collate_ranges)
from eagle.dataset.wrapper import BaseDatasetWrapper
from eagle.dataset.wrapper.utils import cache_and_get_dic_key_type

logger = logging.getLogger("EAGLEDatasetWrapper")


class DatasetWrapperForEAGLE(BaseDatasetWrapper):
    def __init__(
        self,
        dataset: BaseDataset,
        indices: Optional[List[int]] = None,
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
                                                     d_skip_ids=d_skip_ids)
        self.model_name = model_name
        # Check if variables are valid
        assert (
            "neg_doc_ids" not in self.dataset[0] 
            or len(self.dataset[0]["neg_doc_ids"]) == 0
            or len(self.dataset[0]["neg_doc_ids"]) >= nway - 1
        ), f"nway={nway} is larger than the total doc num: {len(self.dataset[0]["neg_doc_ids"])}"
        assert self.model_name == "eagle", f"model_name={model_name} is not supported"

    @property
    def data_keys(self) -> List[str]:
        return ["q_tok_ids", "q_tok_att_mask", "doc_tok_ids", "doc_tok_att_mask", "labels"]

    @property
    def q_phrase_range_key_type(self) -> Union[int, str]:
        return cache_and_get_dic_key_type(cls=self, dic_name="q_phrase_ranges", dic=self.q_phrase_ranges)

    @property
    def d_phrase_range_key_type(self) -> Union[int, str]:
        return cache_and_get_dic_key_type(cls=self, dic_name="d_phrase_ranges", dic=self.d_phrase_ranges)

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
        pids = data["pos_doc_ids"]
        if "neg_doc_ids" in data:
            pids.extend(data["neg_doc_ids"])

        # Extract ranges
        q_phrase_ranges: List[Tuple] = self.q_phrase_ranges[self.q_phrase_range_key_type(qid)]
        d_phrase_ranges: List[List[Tuple]] = [self.d_phrase_ranges[self.d_phrase_range_key_type(i)] for i in pids]

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

        # Combine sentences and replace
        q_tok_ids, q_sent_start_indices = combine_splitted_tok_ids(data["q_tok_ids"])

        ## doc_tok_ids
        d_tok_ids: List[List[int]] = []
        doc_sent_start_indices = []
        for d in data["doc_tok_ids"]:
            tmp_d_tok_ids, tmp_doc_sent_start_indices = combine_splitted_tok_ids(d)
            d_tok_ids.append(tmp_d_tok_ids)
            doc_sent_start_indices.append(tmp_doc_sent_start_indices)

        # Replace the data
        data["q_tok_ids"] = q_tok_ids
        data["doc_tok_ids"] = d_tok_ids

        # Create ones fill with q_tok_ids
        data["q_tok_att_mask"] = torch.ones(len(q_tok_ids), dtype=torch.bool)
        data["doc_tok_att_mask"] = torch.nn.utils.rnn.pad_sequence([torch.ones(len(d), dtype=torch.bool) for d in d_tok_ids], batch_first=True)

        # Add sentence start indices
        data["q_sentence_start_indices"] = q_sent_start_indices
        data["doc_sentence_start_indices"] = doc_sent_start_indices

        # Append sentence mask
        data["q_sent_mask"] = torch.ones(len(q_sent_start_indices), dtype=torch.bool)
        data["doc_sent_mask"] = torch.nn.utils.rnn.pad_sequence([torch.ones(len(d), dtype=torch.bool) for d in doc_sent_start_indices], batch_first=True)

        return data

def combined_phrase_ranges_into_one_sentence(phrase_ranges_list: List[List[Tuple[int, int]]]) -> None:
    SPECIAL_TOK_NUM = 2
    # Figure out the number to add for each sentences
    number_to_adds = [] 
    for sent_idx, p_ranges in enumerate(phrase_ranges_list):    
        if sent_idx == 0 :
            number_to_adds.append(0)
            continue
        # Get the last number to have the cumulative number
        base_number = number_to_adds[-1]
        # Add the max value of the previous sentence
        previous_token_cnt = phrase_ranges_list[sent_idx-1][-1][-1]
        # Remove the first two special tokens if the previous sentence is not the first sentence
        if sent_idx > 0:
            previous_token_cnt = previous_token_cnt - SPECIAL_TOK_NUM
        number_to_adds.append(base_number + previous_token_cnt)

    # Modify the token ranges
    for sent_idx, (number_to_add, p_ranges) in enumerate(zip(number_to_adds, phrase_ranges_list)):
        # Create new p_ranges

        # Remove the ranges for the first two tokens
        if sent_idx != 0:
            p_ranges = p_ranges[SPECIAL_TOK_NUM:]
        # Modify the token idx
        p_ranges = [(s+number_to_add, e+number_to_add) for s, e in p_ranges]

        # update the data
        phrase_ranges_list[sent_idx] =  p_ranges

    # Flatten list of list to list
    phrase_ranges_list = list_utils.do_flatten_list(phrase_ranges_list)

    return phrase_ranges_list

def combine_splitted_tok_ids(tok_ids_list: List[List[int]]) -> Tuple[List[int], List[int]]:
    # This needs to be changed when the adding of the special tokens as prefix changes
    BEGIN_SPECIAL_TOK_NUM = 2
    sent_start_indices = []
    # Remove special tokens in front, except the first sentence
    new_tok_ids_list: List[List[int]] = [tok_ids if idx == 0 else tok_ids[BEGIN_SPECIAL_TOK_NUM:]for idx, tok_ids in enumerate(tok_ids_list)]
    # Combine sentences
    combined_tok_ids = []
    for tok_ids in new_tok_ids_list:
        # Add the start indices
        sent_start_indices.append(len(combined_tok_ids))
        # Add tokens for the current sentence
        combined_tok_ids.extend(tok_ids)

    return tok_ids, sent_start_indices
