import logging
import os
from typing import *

import numpy as np
import torch
from tqdm import tqdm

from colbert.data import Queries, Ranking
from colbert.data.collection import Collection
from colbert.infra.config import ColBERTConfig
from colbert.infra.launcher import print_memory_stats
from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from eagle.model import LightningNewModel
from colbert.noun_extraction.utils import read_in_cache, unidecode_text
from colbert.search.index_storage import IndexScorer

TextQueries = Union[str, "list[str]", "dict[int, str]", Queries]

logger = logging.getLogger("Searcher")


class Searcher:
    def __init__(
        self,
        index,
        checkpoint=None,
        collection=None,
        config=None,
        index_root=None,
        verbose: int = 3,
        skip_loading: bool = False,
        use_cache: bool = False,
    ):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        if index:
            self.index = os.path.join(index_root, index)
            self.index_config = ColBERTConfig.load_from_index(self.index)
        else:
            self.index = None
            self.index_config = None

        # self.checkpoint = checkpoint or self.index_config.checkpoint
        # self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        # self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        if skip_loading:
            self.collection = None
            self.d_phrae_indices = None
        else:
            if use_cache and self.config.is_use_phrase_level:
                file_path = collection or self.config.collection
                logger.info(f"Loading cached parsed data from {file_path}...")
                self.d_phrae_indices = read_in_cache(file_path)
                logger.info(f"Loading done!")
            else:
                self.d_phrae_indices = None
            self.collection = Collection.cast(collection or self.config.collection)
        # self.configure(checkpoint=self.checkpoint, collection=self.collection)

        # self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=verbose)
        self.checkpoint = LightningNewModel.load_from_checkpoint(checkpoint)
        self.checkpoint = self.checkpoint.eval()

        use_gpu = True
        # use_gpu = self.config.total_visible_gpus > 0
        # if use_gpu:
        #     self.checkpoint = self.checkpoint.cuda()
        # load_index_with_mmap = self.config.load_index_with_mmap
        # if load_index_with_mmap and use_gpu:
        # raise ValueError(f"Memory-mapped index can only be used with CPU!")
        if skip_loading:
            self.ranker = None
        else:
            self.ranker = IndexScorer(
                self.index,
                use_gpu,
                load_index_with_mmap=False,
                collection=self.collection,
                d_phrase_indices=self.d_phrae_indices,
                model=self.checkpoint,
                config=config,
            )

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(
        self,
        text: TextQueries,
        full_length_search=False,
        query_toks_to_attend: Optional[List[str]] = None,
        return_valid_embedding: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        queries = text if type(text) is list else [text]
        bsize = 256 if len(queries) > 256 else None

        # result = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search, query_toks_to_attend=query_toks_to_attend, return_valid_embedding=return_valid_embedding)
        q_encoded, q_projected, q_weight, q_scale_factor = self.encode_q_text(
            tok_ids=q_tok_ids,
            att_mask=q_tok_att_mask,
            tok_mask=q_tok_mask,
            scatter_indices=q_scatter_indices,
        )

        return result

    def search(
        self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None
    ):
        Q = self.encode(text, full_length_search=full_length_search)[0]
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
        qid_to_pids=None,
        get_candidates: bool = False,
        return_scores: bool = False,
        required_pids: List[List[int]] = None,
        required_candidates: List[List[int]] = None,
        show_progress: bool = True,
        target_token_indices: Optional[List[Tuple[int]]] = None,
    ):
        queries: Queries = Queries.cast(queries)
        queries_text: List[str] = list(queries.values())
        queries_text = [unidecode_text(q) for q in queries_text]

        # TODO: Phrase-level Similarity
        # Perform tokenization and encoding
        bsize = 512 if len(queries) > 512 else None
        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        logger.info("Tokenizing and encoding queries...")
        if bsize:
            batches = self.checkpoint.query_tokenizer.tensorize(
                queries_text, bsize=bsize, full_length_search=full_length_search
            )
            q_embs = []
            q_weights = []
            q_ids = []
            q_masks = []
            cnt = 0
            for q_ids_, q_masks_ in tqdm(
                batches, desc="Encoding queries", disable=not show_progress
            ):
                q_ids.extend(q_ids_)
                q_masks.extend(q_masks_)
                q_emb, q_weight = self.checkpoint.query(
                    input_ids=q_ids_,
                    attention_mask=q_masks_,
                    input_texts=queries_text[cnt : cnt + len(q_ids_)],
                    target_tokens=(
                        target_token_indices[cnt : cnt + len(q_ids_)]
                        if target_token_indices
                        else None
                    ),
                    is_inference=True,
                )
                q_embs.append(q_emb)
                q_weights.append(q_weight)
                cnt += len(q_ids_)
            q_embs = torch.cat(q_embs, dim=0)
            q_masks = torch.stack(q_masks)
            if q_weights is not None and (
                len(q_weights) > 0 and q_weights[0] is not None
            ):
                q_weights = torch.cat(q_weights, dim=0)
            else:
                q_weights = None
        else:
            q_ids, q_masks = self.checkpoint.query_tokenizer.tensorize(
                queries_text, bsize=bsize, full_length_search=full_length_search
            )
            q_embs, q_weights = self.checkpoint.query(
                input_ids=q_ids,
                input_texts=queries_text,
                attention_mask=q_masks,
                target_tokens=target_token_indices,
                is_inference=True,
            )
        # Perform noun extraction if necessary
        if self.config.is_use_phrase_level:
            logger.info(f"Extracting noun phrases from queries...")
            # Run noun extraction
            phrase_indices_batches = self.checkpoint.get_phrase_indices(
                texts=queries_text,
                tok_ids=q_ids,
                masks=q_masks,
                full_length_search=full_length_search,
            )[0]
            if self.config.is_use_min_threshold:
                noun_phrase_indices_batches = self.checkpoint.get_phrase_indices(
                    texts=queries_text,
                    tok_ids=q_ids,
                    masks=q_masks,
                    full_length_search=full_length_search,
                    prop_noun_only=True,
                )[0]
            else:
                noun_phrase_indices_batches = None
        else:
            phrase_indices_batches = None
            noun_phrase_indices_batches = None

        logger.info(f"Searching top-{k} passages for {len(queries)} queries...")
        results = self._search_all_Q(
            queries,
            Q=q_embs,
            Q_mask=q_masks,
            Q_weights=q_weights,
            k=k,
            q_phrase_indices_batch=phrase_indices_batches,
            q_noun_phrase_indices_batches=noun_phrase_indices_batches,
            filter_fn=filter_fn,
            qid_to_pids=qid_to_pids,
            get_candidates=get_candidates,
            required_pids=required_pids,
            required_candidates=required_candidates,
            return_scores=return_scores,
            show_progress=show_progress,
        )

        return results

    def _search_all_Q(
        self,
        queries: Queries,
        Q: torch.Tensor,
        Q_mask: torch.Tensor,
        Q_weights: torch.Tensor,
        k: int,
        q_phrase_indices_batch: Optional[List[List[Tuple[int, int]]]] = None,
        q_noun_phrase_indices_batches: Optional[List[List[Tuple[int, int]]]] = None,
        filter_fn=None,
        qid_to_pids=None,
        get_candidates: bool = False,
        return_scores: bool = False,
        required_pids: List[List[int]] = None,
        required_candidates: List[List[int]] = None,
        show_progress: bool = True,
    ):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = []
        for query_idx, qid in tqdm(
            enumerate(qids),
            desc="Searching queries",
            total=len(qids),
            disable=not show_progress,
        ):
            result = self.dense_search(
                Q[query_idx : query_idx + 1],
                Q_mask[query_idx],
                Q_weights[query_idx] if Q_weights is not None else None,
                k,
                q_phrase_indices=(
                    q_phrase_indices_batch[query_idx]
                    if q_phrase_indices_batch
                    else None
                ),
                q_noun_phrase_indices=(
                    q_noun_phrase_indices_batches[query_idx]
                    if q_noun_phrase_indices_batches
                    else None
                ),
                filter_fn=filter_fn,
                pids=qid_to_pids[qid],
                required_pids=required_pids[query_idx] if required_pids else None,
                required_candidates=(
                    required_candidates[query_idx] if required_candidates else None
                ),
                get_candidates=get_candidates,
                return_scores=return_scores,
            )
            all_scored_pids.append(result)

        if get_candidates:
            initial_pids, final_pids = zip(*all_scored_pids)
            return initial_pids, final_pids
        else:
            new_d = []
            for i, x in enumerate(all_scored_pids):
                new_d_d = []
                for j, y in enumerate(x):
                    for k, item in enumerate(y):
                        if len(new_d_d) <= k:
                            new_d_d.append([])
                        new_d_d[k].append(item)
                # # Remove those without the same length
                # final_new_d_d = []
                # for item in new_d_d:
                #     if len(item) == len(new_d_d[0]):
                #         final_new_d_d.append(item)
                # new_d_d = final_new_d_d
                new_d.append(new_d_d)
            all_scored_pids = new_d

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = "Searcher::search_all"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(
        self,
        Q: torch.Tensor,
        Q_mask: torch.Tensor,
        Q_weight: Optional[torch.Tensor] = None,
        k=10,
        q_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        q_noun_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        filter_fn=None,
        pids=None,
        required_pids: Optional[List[int]] = None,
        required_candidates: Optional[List[int]] = None,
        get_candidates: bool = False,
        return_scores: bool = False,
    ):
        if k <= 10:
            if self.config.ncells is None or True:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None or True:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None or True:
                self.configure(ndocs=256)
        else:
            if self.config.ncells is None or True:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None or True:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None or True:
                self.configure(ndocs=1024)
            # if self.config.ncells is None or True:
            #     self.configure(ncells=4)
            # if self.config.centroid_score_threshold is None or True:
            #     self.configure(centroid_score_threshold=0.4)
            # if self.config.ndocs is None or True:
            #     self.configure(ndocs=max(k * 4, 4096))

        result = self.ranker.rank(
            self.config,
            Q,
            Q_mask,
            Q_weight,
            q_phrase_indices=q_phrase_indices,
            q_noun_phrase_indices=q_noun_phrase_indices,
            tokenizer=self.checkpoint.doc_tokenizer,
            filter_fn=filter_fn,
            initial_pids=pids,
            required_pids=required_pids,
            required_candidates=required_candidates,
            return_scores=return_scores,
            is_use_min_threshold=self.checkpoint.is_use_min_threshold,
            get_candidates=get_candidates,
        )
        if get_candidates:
            initial_pids, final_pids = result
            return initial_pids, final_pids

        if return_scores:
            all_pids, all_scores, all_token_scores, tokens_below_threshold = result
        else:
            all_pids, all_scores, tokens_below_threshold = result
            all_token_scores = []

        # Get top-k
        pids = all_pids[:k]
        scores = all_scores[:k]
        token_scores = all_token_scores[:k]

        # Append requried pids to the top-k list
        if required_pids is not None:
            if len(token_scores) > 0:
                token_scores = np.split(token_scores, token_scores.shape[0])
            for required_pid in required_pids:
                if required_pid not in pids:
                    gold_idx = all_pids.index(required_pid)
                    pids.append(required_pid)
                    scores.append(all_scores[gold_idx])
                    if len(all_token_scores) > 0:
                        token_scores.append(
                            np.expand_dims(all_token_scores[gold_idx], 0)
                        )
            if len(token_scores) > 0:
                token_scores = np.concatenate(token_scores, axis=0)

        if token_scores is None:
            token_scores = [None] * len(pids)

        return (
            pids,
            list(range(1, len(pids) + 1)),
            scores,
            token_scores,
            [tokens_below_threshold] * len(pids),
        )

    def preprocess_query(self, query: str) -> str:
        query = query.strip()
        if query.startswith(". "):
            query = query[2:]
        if query.endswith("."):
            query = query[:-1]
        if query.endswith("?"):
            query = query[:-1]
        return query.strip()
