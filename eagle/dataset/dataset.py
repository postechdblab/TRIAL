import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import tqdm
from omegaconf import DictConfig
from transformers import AutoTokenizer

from eagle.dataset.utils import preprocess

logger = logging.getLogger("NewDataset")


class RawDataset:
    def __init__(
        self,
        cfg: DictConfig,
        dir_path: str,
        dataset_name: str,
        queries: Union[str, List[Dict]],
        override_nway: int = None,
        is_use_distillation: bool = False,
    ) -> None:
        # Configs
        self.cfg = cfg
        self.nway = cfg.nway if override_nway is None else override_nway
        self.is_debug = cfg.is_debug
        self.sample_size = cfg.sample_size
        self.neg_offset = cfg.negative_start_offset
        self.debug_sample_size = cfg.debug_sample_size
        self.is_use_distillation = is_use_distillation
        # Read in data
        self.data = self.read_data(
            path=os.path.join(
                dir_path,
                dataset_name,
                cfg.distillation_data_file if is_use_distillation else cfg.data_file,
            )
        )
        self.queries = queries
        if type(queries) == str:
            self.queries = self.read_queries(
                path=os.path.join(dir_path, dataset_name, queries)
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, List[str], List[str], Optional[List[float]], Optional[List[float]]]:
        if type(self.data[idx]) == list:
            qid = self.data[idx][0]
            pos_doc_ids = self.data[idx][1]
            neg_doc_ids = self.data[idx][2:]
            # Unzip the pos_doc_id
            if self.is_use_distillation:
                pos_doc_ids, pos_doc_score = pos_doc_ids
                neg_doc_ids, neg_doc_scores = zip(*neg_doc_ids)
            else:
                pos_doc_score = None
                neg_doc_scores = None
            pos_doc_ids = [pos_doc_ids]
        elif type(self.data[idx]) == dict:
            qid = self.data[idx]["id"]
            pos_doc_ids = [item for item in self.data[idx]["answers"]]
            neg_doc_ids = []
            pos_doc_score = None
            neg_doc_scores = None
        # Convert data type
        qid = str(qid)
        pos_doc_ids = [str(item) for item in pos_doc_ids]
        neg_doc_ids = [str(n_id) for n_id in neg_doc_ids]
        return qid, pos_doc_ids, neg_doc_ids, pos_doc_score, neg_doc_scores

    def read_data(self, path: str) -> List[List[int]]:
        data: List = file_utils.read_json_file(path, auto_detect_extension=True)
        # Sample data if needed
        if self.is_debug:
            data = data[: self.debug_sample_size]
        elif self.sample_size > 0:
            data = data[: self.sample_size]
        return data

    def read_queries(self, path: str) -> Dict[str, str]:
        queries: List[Dict] = file_utils.read_json_file(
            path, auto_detect_extension=True
        )
        queries: Dict[str, str] = {query["_id"]: query["text"] for query in queries}
        return queries

    def to_dict(self, corpus: Dict[str, str]) -> Dict:
        # Get negative doc indices
        neg_start_idx = self.neg_offset
        neg_end_idx = neg_start_idx + self.nway - 1

        # Prepare data
        q_ids: List[str] = []
        q_texts: List[str] = []
        pos_doc_texts_list: List[int] = []
        neg_doc_texts_list: List[List[int]] = []
        pos_doc_ids_list: List[str] = []
        neg_doc_ids_list: List[str] = []
        pos_doc_scores: List[float] = []
        neg_doc_scores_list: List[float] = []
        for i, (
            qid,
            pos_doc_ids,
            neg_doc_ids,
            pos_doc_score,
            neg_doc_scores,
        ) in tqdm.tqdm(enumerate(self), desc="Converting to dict", total=len(self)):
            q_ids.append(qid)
            q_texts.append(self.queries[qid])
            pos_doc_texts_list.append(
                [corpus[pos_doc_id] for pos_doc_id in pos_doc_ids]
            )
            for pos_doc_id in pos_doc_ids:
                assert (
                    pos_doc_id not in neg_doc_ids
                ), f"Positive doc id is in negative doc ids: {pos_doc_id} in {neg_doc_ids}"
            neg_doc_texts_list.append(
                [corpus[n_id] for n_id in neg_doc_ids[neg_start_idx:neg_end_idx]]
            )
            pos_doc_ids_list.append(pos_doc_ids)
            neg_doc_ids_list.append(neg_doc_ids[neg_start_idx:neg_end_idx])
            # Append scores if needed
            if pos_doc_score is not None:
                pos_doc_scores.append(pos_doc_score)
            if neg_doc_scores is not None:
                neg_doc_scores_list.append(neg_doc_scores[neg_start_idx:neg_end_idx])

        # Create return dict
        return_dict = {
            "q_ids": q_ids,
            "q_texts": q_texts,
            "pos_doc_texts_list": pos_doc_texts_list,
            "neg_doc_texts_list": neg_doc_texts_list,
            "pos_doc_ids_list": pos_doc_ids_list,
            "neg_doc_ids_list": neg_doc_ids_list,
        }
        # Append scores if needed
        if len(pos_doc_scores) > 0:
            return_dict["pos_doc_scores"] = pos_doc_scores
        if len(neg_doc_scores_list) > 0:
            return_dict["neg_doc_scores_list"] = neg_doc_scores_list

        return return_dict

    def dict_keys(self) -> List[str]:
        """This has to be consistent with the keys in the to_dict method."""
        keys = [
            "q_ids",
            "q_texts",
            "pos_doc_texts_list",
            "neg_doc_texts_list",
            "pos_doc_ids_list",
            "neg_doc_ids_list",
        ]
        if self.is_use_distillation:
            keys += ["pos_doc_scores", "neg_doc_scores_list"]
        return keys


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # Prepare resources
    import functools

    from datasets import Dataset

    from colbert.infra.config.config import ColBERTConfig
    from eagle.dataset.utils import read_corpus

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    my_dataset = RawDataset(
        ColBERTConfig(),
        "/root/ColBERT/data/msmarco/train_data.jsonl",
        "/root/ColBERT/data/msmarco/queries.train.tsv",
    )
    corpus = read_corpus("/root/ColBERT/data/msmarco/corpus.jsonl")
    logger.info("Length of dataset:", len(my_dataset))
    dataset = Dataset.from_dict(my_dataset.to_dict())
    logger.info("to dict complete")
    tokenization = functools.partial(
        preprocess, tokenizer=tokenizer, queries=my_dataset.queries, corpus=corpus
    )
    dataset2 = dataset.map(
        tokenization, batched=True, remove_columns=my_dataset.dict_keys()
    )
    logger.info("Done!")
