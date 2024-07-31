import logging
from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.wrapper import BaseDatasetWrapper
from eagle.model.objective import doc_indices_for_ib_loss

logger = logging.getLogger("DatasetWrapper")


class DatasetWrapperForCrossEncoder(BaseDatasetWrapper):
    def __init__(
        self,
        dataset: BaseDataset,
        indices: Optional[List[int]] = None,
        nway: int = None,
        cache_nway: int = None,
        query_mapping: Dict[int, int] = None,
        corpus_mapping: Dict[int, int] = None,
        q_skip_ids: List[int] = None,
        d_skip_ids: List[int] = None,
    ):
        super(DatasetWrapperForCrossEncoder, self).__init__(
            dataset=dataset,
            indices=indices,
            nway=nway,
            cache_nway=cache_nway,
            query_mapping=query_mapping,
            corpus_mapping=corpus_mapping,
            q_skip_ids=q_skip_ids,
            d_skip_ids=d_skip_ids,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index > len(self):
            raise IndexError(f"Index {index} out of range {len(self)}")
        shuffled_index = self.indices[index]
        selected_data = self.dataset[shuffled_index]

        # Get the correct nway from the cached dataset
        if self.nway < self.cache_nway:
            selected_data["doc_tok_ids"] = selected_data["doc_tok_ids"][: self.nway]
            selected_data["doc_tok_att_mask"] = selected_data["doc_tok_att_mask"][
                : self.nway
            ]
            if "distillation_scores" in selected_data:
                selected_data["distillation_scores"] = selected_data[
                    "distillation_scores"
                ][: self.nway]

        # Extract meta data
        qid = selected_data["q_id"]
        pids = selected_data["pos_doc_ids"]
        if "neg_doc_ids" in selected_data:
            pids.extend(selected_data["neg_doc_ids"])
        pindices = [self.corpus_mapping[pid] for pid in pids]
        pindices = pindices[: self.nway]

        # Get query token ids and attention mask
        q_tok_ids = selected_data["q_tok_ids"]
        q_tok_att_mask = selected_data["q_tok_att_mask"]

        # Get document token ids and attention mask
        doc_tok_ids = selected_data["doc_tok_ids"]
        doc_tok_att_mask = selected_data["doc_tok_att_mask"]

        distillation_scores = selected_data.get("distillation_scores", None)
        labels = selected_data.get("labels", None)

        return {
            "nway": self.nway,
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "labels": labels,
            "distillation_scores": distillation_scores,
        }

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

        bsize = len(input_dics)
        nway = input_dics[0]["nway"]
        ib_nhard = nway // bsize
        repeat_num = ib_nhard * (bsize - 1) + 1

        new_dict = {}
        # Assume all dictionaries have the same keys
        keys = list(input_dics[0].keys())
        # Collate for each key
        for key in keys:
            if input_dics[0][key] is None:
                new_dict[key] = None
                continue
            if key in ["q_tok_ids", "q_tok_att_mask", "labels"]:
                values = [
                    torch.tensor(dic[key], dtype=get_dtype(key), device="cpu")
                    for dic in input_dics
                ]
                padded_values = values
            elif key in ["doc_tok_ids", "doc_tok_att_mask"]:
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
            elif key == "distillation_scores":
                padded_values = torch.nn.functional.log_softmax(
                    torch.tensor(
                        [dic[key] for dic in input_dics],
                        dtype=get_dtype(key),
                        device="cpu",
                    ),
                    dim=-1,
                )
            elif key == "nway":
                continue
            else:
                raise ValueError(f"Unsupported key: {key}")
            new_dict[key] = padded_values

        # Concatenate query and document
        tok_ids = []
        for b_idx in range(bsize):
            q_tok_ids = new_dict["q_tok_ids"][b_idx]
            q_tok_ids = q_tok_ids.unsqueeze(0).repeat_interleave(nway, dim=0)
            tok_ids.append(
                torch.cat([q_tok_ids, new_dict["doc_tok_ids"][b_idx]], dim=1)
            )
        tok_ids = list_utils.do_flatten_list([item.unbind(0) for item in tok_ids])
        tok_ids = pad_sequence(tok_ids, batch_first=True, padding_value=0)
        tok_att_mask = tok_ids != 0

        # Concatenate query and document for in-batch negatives
        d_indices_tensor: torch.Tensor = doc_indices_for_ib_loss(
            bsize, nway, ib_nhard, return_as_tensor=True, device=tok_ids.device
        )
        ib_tok_ids = []
        for b_idx in range(bsize):
            q_tok_ids = new_dict["q_tok_ids"][b_idx]
            q_tok_ids = q_tok_ids.unsqueeze(0).repeat_interleave(repeat_num, dim=0)
            selected_doc_tok_ids = new_dict["doc_tok_ids"][b_idx]

        q_encoded = q_encoded.repeat_interleave(repeat_num, dim=0)
        d_encoded = d_encoded[d_indices_tensor]

        return new_dict
