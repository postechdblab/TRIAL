from typing import *

import hkkang_utils.list as list_utils
import torch
from torch.nn.utils.rnn import pad_sequence

from eagle.model.batch import BaseBatch


class BatchForDPR(BaseBatch):
    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract meta data
        qid = data["q_id"]
        pids = data["pos_doc_ids"]
        if "neg_doc_ids" in data:
            pids.extend(data["neg_doc_ids"])
        pindices = [self.corpus_mapping[pid] for pid in pids]
        pindices = pindices[: self.dataset.nway]

        # Get query token ids and attention mask
        q_tok_ids = data["q_tok_ids"]
        q_tok_att_mask = data["q_tok_att_mask"]

        # Get document token ids and attention mask
        doc_tok_ids = data["doc_tok_ids"]
        doc_tok_att_mask = data["doc_tok_att_mask"]

        distillation_scores = data.get("distillation_scores", None)
        labels = data.get("labels", None)

        return_dic = {
            "nway": self.dataset.nway,
            "q_tok_ids": q_tok_ids,
            "q_tok_att_mask": q_tok_att_mask,
            "doc_tok_ids": doc_tok_ids,
            "doc_tok_att_mask": doc_tok_att_mask,
            "labels": labels,
            "distillation_scores": distillation_scores,
        }

        return return_dic

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
                padded_values = pad_sequence(values, batch_first=True)
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
                torch.cat([q_tok_ids, new_dict["doc_tok_ids"][b_idx][:, 1:]], dim=1)
            )
        tok_ids = list_utils.do_flatten_list([item.unbind(0) for item in tok_ids])
        tok_ids = pad_sequence(tok_ids, batch_first=True, padding_value=0)
        tok_att_mask = tok_ids != 0

        # Concatenate query and document for in-batch negatives
        d_indices_tensor: torch.Tensor = doc_indices_for_ib_loss(
            bsize, nway, ib_nhard, return_as_tensor=True, device=tok_ids.device
        )
        ib_tok_ids = []
        dim = new_dict["doc_tok_ids"].size(-1)
        d_tok_ids = new_dict["doc_tok_ids"].reshape(-1, dim)
        for b_idx in range(bsize):
            q_tok_ids = new_dict["q_tok_ids"][b_idx]
            q_tok_ids = q_tok_ids.unsqueeze(0).repeat_interleave(repeat_num, dim=0)
            # Get document token ids for in-batch negatives
            start_idx = repeat_num * b_idx
            end_idx = repeat_num * (b_idx + 1)
            selected_d_tok_ids = d_tok_ids[d_indices_tensor[start_idx:end_idx]]
            # Concatenate query and document for in-batch negatives
            ib_tok_ids.append(torch.cat([q_tok_ids, selected_d_tok_ids[:, 1:]], dim=1))
        # Pad the ib_tok_ids
        ib_tok_ids = list_utils.do_flatten_list([item.unbind(0) for item in ib_tok_ids])
        ib_tok_ids = pad_sequence(ib_tok_ids, batch_first=True, padding_value=0)
        ib_tok_ids_att_mask = ib_tok_ids != 0

        new_dict["tok_ids"] = tok_ids
        new_dict["tok_att_mask"] = tok_att_mask
        new_dict["ib_tok_ids"] = ib_tok_ids
        new_dict["ib_tok_att_mask"] = ib_tok_ids_att_mask

        return new_dict
