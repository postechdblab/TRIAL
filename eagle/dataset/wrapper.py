import logging
from typing import *

from eagle.dataset.base_dataset import BaseDataset
from eagle.dataset.utils import (add_doc_ranges_and_mask,
                                 add_query_ranges_and_mask, is_token_included)

logger = logging.getLogger("DatasetWrapper")


class DatasetWrapper:
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
        granularity_level: Optional[str] = None,
        is_use_fine_grained_loss: Optional[bool] = False,
    ):
        self.dataset = dataset
        self.indices = [i for i in range(len(dataset))] if indices is None else indices
        self.nway = nway
        self.cache_nway = cache_nway
        self.q_word_ranges = q_word_ranges
        self.q_phrase_ranges = q_phrase_ranges
        self.d_word_ranges = d_word_ranges
        self.d_phrase_ranges = d_phrase_ranges
        self.corpus_mapping = corpus_mapping
        self.query_mapping = query_mapping
        self.q_skip_ids = q_skip_ids
        self.d_skip_ids = d_skip_ids
        self.granularity_level = granularity_level
        self.is_use_fine_grained_loss = is_use_fine_grained_loss
        # Check if variables are valid
        assert len(self.dataset) == len(
            self.indices
        ), f"len(self.dataset)={len(self.dataset)}, len(self.indices)={len(self.indices)}"
        assert nway is not None, f"nway is None. Please provide nway."
        assert cache_nway is not None, f"cache_nway is None. Please provide cache_nway."
        assert cache_nway >= nway, f"cache_nway={cache_nway}, nway={nway}"
        assert (
            "neg_doc_ids" not in self.dataset[0] 
            or len(self.dataset[0]["neg_doc_ids"]) == 0
            or len(self.dataset[0]["neg_doc_ids"]) >= nway - 1
        ), f"nway={nway} is larger than the total doc num: {len(self.dataset[0]["neg_doc_ids"])}"
        assert self.granularity_level in ["token", "word", "phrase"]
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

    @property
    def is_use_multi_granularity(self) -> bool:
        return self.granularity_level in ["word", "phrase"]

    def __getitem__(self, idx: int) -> Dict:
        # Replace the nway
        if idx >= len(self):
            logger.info(
                f"idx={idx}, len(self)={len(self)}, len(self.indices)={len(self.indices)}"
            )
        if idx >= len(self.indices):
            logger.info(
                f"idx={idx}, len(self)={len(self)}, len(self.indices)={len(self.indices)}"
            )
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
        q_word_ranges: List[Tuple] = []
        q_phrase_ranges: List[Tuple] = []
        d_word_ranges: List[List[Tuple]] = [[] for _ in range(len(pindices))]
        d_phrase_ranges: List[List[Tuple]] = [[] for _ in range(len(pindices))]
        if self.granularity_level == "token":
            q_word_ranges = [(i, i + 1) for i in range(len(data["q_tok_ids"]))]
            d_word_ranges = [
                [(i, i + 1) for i in range(len(item))] for item in data["doc_tok_ids"]
            ]
        elif self.granularity_level in ["word"]:
            q_word_ranges = self.q_word_ranges[qidx]
            d_word_ranges = [self.d_word_ranges[i] for i in pindices]
        elif self.granularity_level in ["phrase"]:
            q_phrase_ranges = self.q_phrase_ranges[qidx]
            d_phrase_ranges = [self.d_phrase_ranges[i] for i in pindices]

        # Add ranges and token mask
        data = add_query_ranges_and_mask(
            input_dict=data,
            word_ranges=q_word_ranges,
            phrase_ranges=q_phrase_ranges,
            skip_ids=self.q_skip_ids,
            use_coarse_emb=self.is_use_multi_granularity,
        )
        data = add_doc_ranges_and_mask(
            input_dict=data,
            word_ranges=d_word_ranges,
            phrase_ranges=d_phrase_ranges,
            skip_ids=self.d_skip_ids,
            use_coarse_emb=self.is_use_multi_granularity,
        )
        # Add index for positive document
        data["pos_doc_idxs"] = [self.corpus_mapping[i] for i in data["pos_doc_ids"]]

        # Add fine-grained label
        if self.is_use_fine_grained_loss:
            target = data["q_tok_ids"]
            data["fine_grained_label"] = [
                is_token_included(src=item, target=target)
                for item in data["doc_tok_ids"][1:]
            ]

        return data

    def __len__(self) -> int:
        return len(self.dataset)
