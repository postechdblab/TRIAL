import os
import pathlib
from math import ceil
from typing import *

import torch
from torch.utils.cpp_extension import load

from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)
from colbert.modeling.colbert import (
    ColBERT,
    colbert_score,
    colbert_score_packed,
    colbert_score_phrase_level,
    colbert_score_reduce,
)
from colbert.modeling.tokenization.utils import get_phrase_indices
from colbert.noun_extraction.identify_noun import SpacyModel
from colbert.noun_extraction.utils import (
    modify_indices_for_special_tokens,
    rearrange_indices,
    unidecode_text,
)
from colbert.search.candidate_generation import CandidateGeneration
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message

from .index_loader import IndexLoader


class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(
        self,
        index_path,
        use_gpu=True,
        load_index_with_mmap=False,
        collection=None,
        d_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        model: ColBERT = None,
        config: Optional[Any] = None,
    ):
        super().__init__(
            index_path=index_path,
            use_gpu=use_gpu,
            load_index_with_mmap=load_index_with_mmap,
        )

        IndexScorer.try_load_torch_extensions(use_gpu)

        self.set_embeddings_strided()
        self.collection = collection
        self.d_phrase_indices = d_phrase_indices
        self.model = model
        self.config_ = config

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(
            f"Loading filter_pids_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        filter_pids_cpp = load(
            name="filter_pids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "filter_pids.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.filter_pids = filter_pids_cpp.filter_pids_cpp

        print_message(
            f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        cls.loaded_extensions = True

    def set_embeddings_strided(self):
        if self.load_index_with_mmap:
            assert self.num_chunks == 1
            self.offsets = torch.cumsum(self.doclens, dim=0)
            self.offsets = torch.cat((torch.zeros(1, dtype=torch.int64), self.offsets))
        else:
            self.embeddings_strided = ResidualEmbeddingsStrided(
                self.codec, self.embeddings, self.doclens
            )
            self.offsets = self.embeddings_strided.codes_strided.offsets

    def lookup_pids(self, passage_ids, out_device="cuda", return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(
        self, config, Q: torch.Tensor, Q_mask: Optional[torch.Tensor]
    ) -> Tuple:
        # Q = Q[:, :config.query_maxlen]   # NOTE: Candidate generation uses only the query tokens
        pids, centroid_scores = self.generate_candidates(config, Q, Q_mask)

        return pids, centroid_scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(
        self,
        config,
        Q: torch.Tensor,
        Q_mask: torch.Tensor,
        Q_weights: Optional[torch.Tensor] = None,
        q_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        q_noun_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        tokenizer: Optional[Any] = None,
        filter_fn=None,
        query_tok_ids: Optional[List[int]] = None,
        initial_pids=None,
        required_pids: Optional[List[int]] = None,
        required_candidates: Optional[List[int]] = None,
        return_scores: bool = False,
        is_use_min_threshold: bool = False,
        get_candidates: bool = False,
    ):
        with torch.inference_mode():
            if initial_pids is None:
                initial_pids, centroid_scores = self.retrieve(config, Q, Q_mask)
            else:
                pids_, centroid_scores = self.retrieve(config, Q, Q_mask)
                initial_pids = torch.tensor(
                    initial_pids, dtype=pids_.dtype, device=pids_.device
                )

            if filter_fn is not None:
                filtered_pids = filter_fn(initial_pids)
                assert isinstance(filtered_pids, torch.Tensor), type(filtered_pids)
                assert (
                    filtered_pids.dtype == pids.dtype
                ), f"filtered_pids.dtype={filtered_pids.dtype}, pids.dtype={initial_pids.dtype}"
                assert (
                    filtered_pids.device == pids.device
                ), f"filtered_pids.device={filtered_pids.device}, pids.device={initial_pids.device}"
                initial_pids = filtered_pids
                if len(initial_pids) == 0:
                    return [], []

            results = self.score_pids(
                config,
                Q,
                Q_weights,
                initial_pids,
                centroid_scores,
                q_phrase_indices=q_phrase_indices,
                tokenizer=tokenizer,
                required_pids=required_pids,
                required_candidates=required_candidates,
                get_candidates=get_candidates,
                return_scores=return_scores,
            )

            if get_candidates:
                return initial_pids.tolist(), results[1]

            if return_scores:
                scores, pids, token_scores = results
                # scores, top_indices, token_scores = scores, pids
            else:
                scores, pids = results
                token_scores = None

            # Sum over all query tokens
            if is_use_min_threshold:
                all_scores = scores.new_tensor(scores)
                all_scores = scores.clone().detach()
            scores = scores.sum(dim=1)
            scores_sorter = scores.sort(descending=True)

            # Check if
            min_threshold = 0.5
            examine_top_k = 10
            tokens_below_threshold: List[Tuple[int, int]] = []
            if is_use_min_threshold:
                # Noun phrase with scores lower than the threshold
                all_scores = all_scores[scores_sorter.indices[:examine_top_k]]
                all_scores = all_scores.transpose(0, 1)
                for i, score in enumerate(all_scores):
                    if q_phrase_indices[i] in q_noun_phrase_indices:
                        num_words_in_phrase = (
                            q_phrase_indices[i][1] - q_phrase_indices[i][0]
                        )
                        min_score = torch.min(score).item() / num_words_in_phrase
                        if min_score < min_threshold:
                            tokens_below_threshold.append(q_phrase_indices[i])
            # TODO:
            if is_use_min_threshold and tokens_below_threshold:
                Q_mask = torch.zeros_like(Q, dtype=torch.float32)
                for start_idx, end_idx in tokens_below_threshold:
                    Q_mask[:, start_idx:end_idx] = 1
                Q = Q * Q_mask
                new_scores, new_pids, new_token_scores = self.score_pids(
                    config,
                    Q,
                    Q_weights,
                    initial_pids,
                    centroid_scores,
                    q_phrase_indices=q_phrase_indices,
                    tokenizer=tokenizer,
                    required_pids=required_pids,
                    get_candidates=get_candidates,
                    return_scores=return_scores,
                )
                new_scores = new_scores.sum(dim=1)
                new_scores_sorter = new_scores.sort(descending=True)
                final_pids = []
                final_scores = []
                final_token_scores = []
                for i in range(0, len(new_scores_sorter.indices) // 5, 5):
                    for j in range(i, i + 5):
                        idx = scores_sorter.indices[j]
                        if idx not in final_pids:
                            final_pids.append(pids[idx].item())
                            final_scores.append(scores[idx].item())
                            if return_scores:
                                final_token_scores.append(token_scores[idx].tolist())
                    for j in range(i, i + 5):
                        idx = new_scores_sorter.indices[j]
                        if new_pids[idx] not in final_pids:
                            final_pids.append(new_pids[idx].item())
                            final_scores.append(new_scores[idx].item())
                            if return_scores:
                                final_token_scores.append(
                                    new_token_scores[idx].tolist()
                                )
            # Sort the passages based on the phrase scores...
            else:
                pids, scores = (
                    pids[scores_sorter.indices].tolist(),
                    scores_sorter.values.cpu().tolist(),
                )
                if token_scores is not None:
                    token_scores = token_scores[scores_sorter.indices].cpu().numpy()
            if return_scores:
                return pids, scores, token_scores, tokens_below_threshold
            return pids, scores, tokens_below_threshold

    def score_pids(
        self,
        config,
        Q,
        Q_weights,
        pids,
        centroid_scores,
        q_phrase_indices: Optional[List[Tuple[int, int]]] = None,
        tokenizer: Optional[Any] = None,
        required_pids: Optional[List[int]] = None,
        required_candidates: Optional[List[int]] = None,
        get_candidates: bool = False,
        return_scores: bool = False,
    ):
        """
        Always supply a flat list or tensor for `pids`.

        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.
        """

        # TODO: Remove batching?
        batch_size = 2**20

        if self.use_gpu:
            centroid_scores = centroid_scores.cuda()

        idx = centroid_scores.max(-1).values >= config.centroid_score_threshold

        if self.use_gpu:
            approx_scores = []

            skip_plaid_stage2 = False
            if not skip_plaid_stage2:
                # Filter docs using pruned centroid scores
                for i in range(0, ceil(len(pids) / batch_size)):
                    pids_ = pids[i * batch_size : (i + 1) * batch_size]
                    codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(
                        pids_
                    )
                    idx_ = idx[codes_packed.long()]
                    pruned_codes_strided = StridedTensor(
                        idx_, codes_lengths, use_gpu=self.use_gpu
                    )
                    pruned_codes_padded, pruned_codes_mask = (
                        pruned_codes_strided.as_padded_tensor()
                    )
                    pruned_codes_lengths = (
                        pruned_codes_padded * pruned_codes_mask
                    ).sum(dim=1)
                    codes_packed_ = codes_packed[idx_]
                    approx_scores_ = centroid_scores[codes_packed_.long()]
                    if approx_scores_.shape[0] == 0:
                        approx_scores.append(
                            torch.zeros(
                                (len(pids_),), dtype=approx_scores_.dtype
                            ).cuda()
                        )
                        continue
                    approx_scores_strided = StridedTensor(
                        approx_scores_, pruned_codes_lengths, use_gpu=self.use_gpu
                    )
                    approx_scores_padded, approx_scores_mask = (
                        approx_scores_strided.as_padded_tensor()
                    )
                    approx_scores_ = colbert_score_reduce(
                        approx_scores_padded, approx_scores_mask
                    )
                    approx_scores.append(approx_scores_)
                approx_scores = torch.cat(approx_scores, dim=0)
                assert approx_scores.is_cuda, approx_scores.device
                if config.ndocs < len(approx_scores):
                    pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]
            # Filter docs using full centroid scores
            codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(pids)
            approx_scores = centroid_scores[codes_packed.long()]
            approx_scores_strided = StridedTensor(
                approx_scores, codes_lengths, use_gpu=self.use_gpu
            )
            approx_scores_padded, approx_scores_mask = (
                approx_scores_strided.as_padded_tensor()
            )
            approx_scores = colbert_score_reduce(
                approx_scores_padded, approx_scores_mask
            )
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
        else:
            pids = IndexScorer.filter_pids(
                pids,
                centroid_scores,
                self.embeddings.codes,
                self.doclens,
                self.offsets,
                idx,
                config.ndocs,
            )

        # Include the required passage, if any
        if required_pids is not None or required_candidates is not None:
            pids_list = pids.tolist()
            pids_to_include = []
            if required_pids is not None:
                pids_to_include = pids_to_include + list(
                    filter(lambda pid: pid not in pids_list, required_pids)
                )
            if required_candidates is not None:
                pids_to_include = pids_to_include + list(
                    filter(lambda pid: pid not in pids_list, required_candidates)
                )
            pids = torch.tensor(
                pids_to_include + pids_list, dtype=pids.dtype, device=pids.device
            )

        if get_candidates:
            return None, pids.tolist()

        # Remove the pid 0 (i.e., dummy pid) if it is in the list ()
        if 0 in pids.tolist():
            pids = pids[pids != 0]

        # Rank final list of docs using full approximate embeddings (including residuals)
        if self.use_gpu:
            D_packed, D_mask = self.lookup_pids(pids)
        else:
            D_packed = IndexScorer.decompress_residuals(
                pids,
                self.doclens,
                self.offsets,
                self.codec.bucket_weights,
                self.codec.reversed_bit_map,
                self.codec.decompression_lookup_table,
                self.embeddings.residuals,
                self.embeddings.codes,
                self.codec.centroids,
                self.codec.dim,
                self.codec.nbits,
            )
            D_packed = torch.nn.functional.normalize(
                D_packed.to(torch.float32), p=2, dim=-1
            )
            D_mask = self.doclens[pids.long()]

        tokenwise_scores = None
        if Q.size(0) == 1:
            if self.config_.is_use_phrase_level:
                # Get d_phrase_indices
                if self.d_phrase_indices:
                    d_phrase_indices_batch = []
                    for pid in pids.tolist():
                        d_phrase_indices_batch.append(self.d_phrase_indices[pid])
                    # TODO: Get mask by other ways for efficiency
                    docs = [
                        unidecode_text(self.collection[pid]) for pid in pids.tolist()
                    ]
                    # Tokenize texts
                    doc_ids, doc_masks = tokenizer.tensorize(docs, bsize=len(docs))[0][
                        0
                    ]
                    masks = (
                        torch.tensor(
                            self.model.mask(doc_ids, skiplist=self.model.skiplist_ids),
                            device=doc_ids.device,
                        )
                        .unsqueeze(2)
                        .float()
                    )
                    # Rearrange indices for cls and unused token in front (i.e., adding 2)
                    d_phrase_indices_batch = [
                        modify_indices_for_special_tokens(item, append_suffix=False)
                        for item in d_phrase_indices_batch
                    ]
                else:
                    # Get documents from collection using pids
                    docs = [unidecode_text(self.collection[pid]) for pid in pids]
                    # Tokenize texts
                    doc_ids, doc_masks = tokenizer.tensorize(docs, bsize=len(docs))[0][
                        0
                    ]
                    masks = (
                        torch.tensor(
                            self.model.mask(doc_ids, skiplist=self.model.skiplist_ids),
                            device=doc_ids.device,
                        )
                        .unsqueeze(2)
                        .float()
                    )
                    # Run noun extraction
                    d_parsed_docs: List[Text] = SpacyModel()(
                        docs, max_token_num=config.doc_maxlen
                    )
                    # Extract all phrase indices
                    d_phrase_indices_batch = get_phrase_indices(
                        tok_ids=doc_ids,
                        masks=doc_masks,
                        tokenizer=tokenizer.tok,
                        input_texts=docs,
                        parsed_texts=d_parsed_docs,
                        bsize=len(docs),
                    )[0]
                    # Remove tokens with mask 0
                    doc_ids = [
                        filter_tokens(doc_id, mask)
                        for doc_id, mask in zip(doc_ids, masks)
                    ]

                # Remove d_phrase_indices that exceed the max length
                new_d_phrase_indices_batch = []
                for i, d_phrase_indices in enumerate(d_phrase_indices_batch):
                    new_d_phrase_indices = []
                    for start_idx, end_idx in d_phrase_indices:
                        if end_idx <= config.doc_maxlen:
                            new_d_phrase_indices.append((start_idx, end_idx))
                        elif start_idx < config.doc_maxlen:
                            new_d_phrase_indices.append((start_idx, config.doc_maxlen))
                    new_d_phrase_indices_batch.append(new_d_phrase_indices)
                d_phrase_indices_batch = new_d_phrase_indices_batch

                if self.d_phrase_indices:
                    # Append the suffix
                    for d_phrase_indices in d_phrase_indices_batch:
                        last_start_idx, last_end_idx = d_phrase_indices[-1]
                        if last_end_idx == config.doc_maxlen:
                            if last_start_idx + 1 == last_end_idx:
                                pass
                            else:
                                d_phrase_indices[-1] = (
                                    last_start_idx,
                                    last_end_idx - 1,
                                )
                                d_phrase_indices.append(
                                    (last_end_idx - 1, last_end_idx)
                                )
                        else:
                            d_phrase_indices.append((last_end_idx, last_end_idx + 1))

                # Modify d_phrase_indices_batch by removing the skipped tokens
                # TODO: Need to check if this is correct
                d_phrase_indices_batch = rearrange_indices(
                    phrase_indices_batch=d_phrase_indices_batch, final_mask=masks
                )

                # Add phrases for padding
                if not self.config_.is_skip_padding and q_phrase_indices:
                    # Fill in the phrase indices for query
                    max_q_len = config.query_maxlen
                    last_idx = q_phrase_indices[-1][-1]
                    for i in range(last_idx, max_q_len):
                        q_phrase_indices.append((i, i + 1))
                # Compute scores
                scores = colbert_score_phrase_level(
                    Q=Q,
                    D_packed=D_packed,
                    D_lengths=D_mask,
                    q_phrase_indices=[q_phrase_indices],
                    d_phrase_indices_batch=d_phrase_indices_batch,
                    return_scores=return_scores,
                )
            else:
                # Compute scores
                if Q_weights is not None:
                    Q = Q * Q_weights.unsqueeze(-1)
                scores = colbert_score_packed(
                    Q, D_packed, D_mask, return_scores=return_scores
                )
            if return_scores:
                scores, tokenwise_scores = scores
        else:
            D_strided = StridedTensor(D_packed, D_mask, use_gpu=self.use_gpu)
            D_padded, D_lengths = D_strided.as_padded_tensor()
            if Q_weights is not None:
                Q = Q * Q_weights.unsqueeze(1)
            scores = colbert_score(Q, D_padded, D_lengths, return_scores=return_scores)

        if tokenwise_scores is not None:
            tokenwise_scores = tokenwise_scores.transpose(1, 2)

        if return_scores:
            return scores, pids, tokenwise_scores
        return scores, pids


def filter_tokens(doc_ids, doc_masks):
    """
    Removes tokens from doc_ids where the corresponding value in doc_masks is 0.

    Parameters:
    - doc_ids: List of token IDs.
    - doc_masks: List of mask values corresponding to each token in doc_ids.

    Returns:
    - Filtered list of token IDs where their corresponding mask value is not 0.
    """
    # Ensure doc_ids and doc_masks are of the same length
    if len(doc_ids) != len(doc_masks):
        raise ValueError("doc_ids and doc_masks must be of the same length")

    # Filter out tokens where mask value is 0
    filtered_doc_ids = [doc_id for doc_id, mask in zip(doc_ids, doc_masks) if mask != 0]

    return torch.stack(filtered_doc_ids)
