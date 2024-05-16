import argparse
import copy
import logging
from typing import *

import hkkang_utils.file as file_utils
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from nltk.corpus import stopwords

from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.utils import get_phrase_indices
from colbert.noun_extraction.identify_noun import SpacyModel, Text
from colbert.noun_extraction.utils import unidecode_text
from colbert.utils.utils import stem
from model import RetrievalResult
from model.utils import Document
from scripts.evaluate.utils import load_data

logger = logging.getLogger("TokenScore")

DATASET_DIR = "/root/ColBERT/data"

# Possible keys
TYPE_KEYS = ["all", "noun", "prop_noun", "stop", "special", "others"]
OUTER_KEYS = ["all", "correct", "incorrect"]
INNER_KEYS = ["all", "pos", "neg"]


def index_translation_mapping(mask: List[str]) -> List[str]:
    cnt = 0
    mapping: Dict[int, int] = {}
    for i in range(len(mask)):
        if mask[i] == 1:
            mapping[i] = cnt
            cnt += 1
    # Reverse the mapping
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping


def get_max_doc_tokens(
    result: RetrievalResult, tokenizer, return_all_ids: bool = False
) -> Tuple[List[int], List[int]]:
    """Return the indices and token ids for the document tokens that maps to the max score for each query token."""
    d_ids = tokenizer.tensorize([result.doc.unidecoded_text_only])[0][0].tolist()
    d_mask = tokenizer.tensorize_packed([result.doc.unidecoded_text_only])[1]
    # Find the max score and index for each token
    max_indices = result.token_scores.argmax(axis=1).tolist()
    mapping = index_translation_mapping(mask=d_mask.tolist())
    flatted_idx = [mapping[i] for i in max_indices]
    if return_all_ids:
        return flatted_idx, [d_ids[i] for i in flatted_idx], d_ids
    return flatted_idx, [d_ids[i] for i in flatted_idx]


def get_orginal_word(idx: int, toks: List[str]) -> str:
    """Return the original word from the word piece tokens."""
    words = []
    if toks[idx].startswith("##"):
        # Append the previous tokens
        j = idx - 1
        while j >= 0:
            if toks[j].startswith("##"):
                words.insert(0, toks[j].replace("##", ""))
                j -= 1
            else:
                words.insert(0, toks[j])
                break
        # Append the current token
        words.append(toks[idx].replace("##", ""))
        # Append the next tokens
        j = idx + 1
        while j < len(toks) and toks[j].startswith("##"):
            words.append(toks[j].replace("##", ""))
            j += 1
    else:
        words.append(toks[idx])
        j = idx + 1
        while j < len(toks) and toks[j].startswith("##"):
            words.append(toks[j].replace("##", ""))
            j += 1
    return "".join(words)


def is_q_d_token_match(
    q_toks,
    d_toks,
    q_start_idx,
    q_end_idx,
    d_max_token_idx,
    tag: Optional[str] = "",
    print_false: bool = False,
) -> bool:
    is_match = True
    q_word = "".join(q_toks[q_start_idx : q_end_idx + 1]).replace("##", "")
    q_word = stem(q_word)
    for i in d_max_token_idx[q_start_idx : q_end_idx + 1]:
        d_original_word = get_orginal_word(i, d_toks)
        d_original_word = stem(d_original_word)
        if d_original_word not in q_word and q_word not in d_original_word:
            is_match = False
            if print_false:
                print(f"Mismatch ({tag}): {q_word} != {d_original_word}")
            break

    return is_match


def is_q_d_phrase_match(
    q_toks, d_toks, q_start_idx, q_end_idx, d_max_token_idx
) -> bool:
    q_words = []
    for i in range(q_start_idx, q_end_idx + 1):
        if q_toks[i].startswith("##"):
            if q_words:
                q_words[-1] = q_words[-1] + q_toks[i].replace("##", "")
            else:
                q_words.append(q_toks[i].replace("##", ""))
        else:
            q_words.append(q_toks[i])
    q_words = [stem(word) for word in q_words]
    q_word = " ".join(q_words)
    d_max_token_idx = d_max_token_idx[q_start_idx : q_end_idx + 1]
    d_words = []
    for i in d_max_token_idx:
        if d_toks[i].startswith("##"):
            if d_words:
                d_words[-1] = d_words[-1] + d_toks[i].replace("##", "")
            else:
                d_words.append(d_toks[i].replace("##", ""))
        else:
            d_words.append(d_toks[i])
    d_words = [stem(word) for word in d_words]
    d_word = " ".join(d_words)
    return q_word == d_word


def get_result_score(result) -> float:
    return sum(result.token_scores.max(axis=1))


def get_word_piece_indices(toks: List[str]) -> List[Tuple[int, int]]:
    stack: List[int] = []
    indices: List[Tuple[int, int]] = []
    for i, tok in enumerate(toks):
        if tok.startswith("##"):
            if len(stack) == 0:
                stack.append(i - 1)
            stack.append(i)
        else:
            # Save the stack and reset
            if stack:
                indices.append((stack[0], stack[-1]))
                stack = []
    # Save the last stack
    if stack:
        indices.append((stack[0], stack[-1]))
    return indices


def main(
    dataset_name: str,
    result_file_path: str,
    q_max_len: int,
    d_max_len: int,
    eval_type: str = None,
    for_copy: bool = False,
) -> None:
    """Load the evaluation results from the cache and analyze the token scores."""
    # Settings
    do_doc_tokens = False
    compare_token_between_pos_neg = True
    show_token_type_difference = False
    show_granularity_problem = True
    only_negative_for_rescoring = False

    # Load tokenizer
    config = ColBERTConfig()
    config.query_maxlen = q_max_len
    config.doc_maxlen = d_max_len
    config.checkpoint = "bert-base-uncased"
    q_tokenizer = QueryTokenizer(config)
    d_tokenizer = DocTokenizer(config)
    # Get stop words
    stop_words = set(stopwords.words("english"))
    stop_word_ids = [
        q_tokenizer.tok.convert_tokens_to_ids(q_tokenizer.tok.tokenize(word))
        for word in stop_words
    ]

    # Load result data
    logger.info(f"Reading results cache data")
    result_data: List = file_utils.read_pickle_file(result_file_path)
    # Load dev data
    logger.info(f"Loading {dataset_name} data")
    eval_data: List = load_data(dataset_dir=DATASET_DIR, dataset_name=dataset_name)

    assert len(result_data) == len(
        eval_data
    ), f"Length mismatch: {len(result_data)} != {len(eval_data)}"
    logger.info(f"Loaded {len(result_data)} data")

    # Type of scores to collect
    scores_dict: Dict[str, Dict[str, List]] = dict()
    # Initialize the scores list
    for tk in TYPE_KEYS:
        for ok in OUTER_KEYS:
            for ik in INNER_KEYS:
                if tk not in scores_dict:
                    scores_dict[tk] = dict()
                scores_dict[tk][f"{ok}_{ik}"] = []

    is_correct_cnt = 0
    num_negative_results_when_incorrect = 0
    cnt_q_with_phrases_of_multi_word = 0
    stat = {
        "no_phrase": {
            "correct": 0,
            "incorrect": 0,
        },
        "phrase": {
            "correct": 0,
            "incorrect": 0,
        },
    }
    only_pos_match = 0
    only_neg_match = 0
    only_pos_match_phrase = 0
    only_neg_match_phrase = 0
    for q_i, datum in enumerate(tqdm.tqdm(eval_data, desc="Analzying data")):
        qid, query, gold_pids, gold_p_titles, gold_p_scores = datum
        query = unidecode_text(query)
        # Get current retrieval results
        gold_pids_, pos_results, all_top_results = result_data[qid]
        pos_results: List[RetrievalResult] = pos_results
        top_results: List[RetrievalResult] = all_top_results
        top_k_reusults: List[RetrievalResult] = top_results[:10]
        # Check pos_results
        assert all(
            [str(item.doc.id) in gold_pids for item in pos_results]
        ), f"Positive results mismatch: {pos_results} != {gold_pids}"
        assert set(gold_pids_) == set(
            gold_pids
        ), f"Gold pids mismatch: {gold_pids_} != {gold_pids}"
        # Get neg_results
        neg_results = [
            item for item in top_results if str(item.doc.id) not in gold_pids
        ]
        min_pos_score = min([get_result_score(item) for item in pos_results])
        hard_neg_results = [
            item for item in neg_results if get_result_score(item) >= min_pos_score
        ]
        # neg_result_to_remove = neg_results[0]
        # neg_results = [item for item in neg_results if item.doc.id != neg_result_to_remove.doc.id]
        neg_pids = [str(item.doc.id) for item in neg_results]
        # num_negative_results += len(neg_results)
        # num_positive_results += len(pos_results)
        # Get all results
        all_doc_ids = []
        all_results: List[RetrievalResult] = []
        for item in pos_results + hard_neg_results:
            assert item.doc.id not in all_doc_ids, f"Duplicate doc id: {item.doc.id}"
            all_results.append(item)
            all_doc_ids.append(item.doc.id)
        # Tokenize query
        q_ids, q_mask = q_tokenizer.tensorize([query], bsize=1)[0]
        q_toks = q_tokenizer.tok.convert_ids_to_tokens(q_ids[0])
        # Figure out the phrase indices
        parsed_q: List[Text] = SpacyModel()([query])
        q_phrases_prop = get_phrase_indices(
            q_ids,
            q_mask,
            q_tokenizer.tok,
            [query],
            parsed_q,
            bsize=1,
            prop_noun_only=True,
        )[0][0]
        q_prop_noun_indices = [i for s, e in q_phrases_prop for i in range(s, e)]
        q_phrases_noun = get_phrase_indices(
            q_ids, q_mask, q_tokenizer.tok, [query], parsed_q, bsize=1, noun_only=True
        )[0][0]
        q_noun_indices = [i for s, e in q_phrases_noun for i in range(s, e)]
        q_stop_indices = []
        i = 0
        q_ids = q_ids.squeeze(0).tolist()
        while i < len(q_toks):
            for stop_ids in stop_word_ids:
                stop_len = len(stop_ids)
                if i + stop_len >= len(q_toks):
                    continue
                if q_ids[i : i + stop_len] == stop_ids:
                    q_stop_indices.extend(list(range(i, i + stop_len)))
                    i += stop_len - 1
                    break
            i += 1
        q_special_indices = [0, 1, len(q_ids) - 1]
        found_indices = set(
            q_noun_indices + q_prop_noun_indices + q_stop_indices + q_special_indices
        )
        q_others_indices = [i for i in range(len(q_toks)) if i not in found_indices]
        # Indices for tokens of word pices
        q_word_indices: List[List[int]] = get_word_piece_indices(q_toks)
        # Check if the gold pids are all found within top-10
        retrieved_doc_ids = [str(item.doc.id) for item in top_k_reusults]
        if eval_type == "all":
            eval_func = all
        elif eval_type == "any":
            eval_func = any
        else:
            raise ValueError(f"Invalid eval_type: {eval_type}")
        if len(gold_pids) > 10:
            is_correct = eval_func([p in gold_pids for p in retrieved_doc_ids])
        else:
            is_correct = eval_func([p in retrieved_doc_ids for p in gold_pids])

        if is_correct:
            is_correct_cnt += 1
        else:
            num_negative_results_when_incorrect += len(neg_results)

        inner_key = "correct" if is_correct else "incorrect"
        # Get query phrase stats
        # Initialize tokenizer
        # Load tokenizer
        logger.info(f"Getting query phrase stats...")
        parsed_texts: List[Text] = SpacyModel()(
            [query], max_token_num=config.query_maxlen
        )
        input_ids, attention_mask = q_tokenizer.tensorize([query], bsize=len([query]))[
            0
        ]
        q_phrases = get_phrase_indices(
            input_ids,
            attention_mask,
            q_tokenizer.tok,
            [query],
            parsed_texts,
            bsize=len([query]),
            all_noun_only=True,
        )[0][0]
        #
        filtered_items = [j - i for i, j in q_phrases if j - i > 1]
        if filtered_items and any([item > 1 for item in filtered_items]):
            outer_key = "phrase"
            cnt_q_with_phrases_of_multi_word += 1
        else:
            outer_key = "no_phrase"
        stat[outer_key][inner_key] += 1

        # Check q-d match for pos with phrases
        if q_phrases:
            pos_is_match = []
            for result in pos_results:
                d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                    result, d_tokenizer, return_all_ids=True
                )
                d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
                d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
                has_match = False
                item_found = False
                for s, e in q_phrases:
                    if e - s == 1:
                        continue
                    # Check if is match
                    item_found = True
                    is_match = is_q_d_phrase_match(
                        q_toks=q_toks,
                        d_toks=d_toks,
                        q_start_idx=s,
                        q_end_idx=e,
                        d_max_token_idx=d_max_token_idx,
                    )
                    has_match = has_match or is_match
                if item_found:
                    pos_is_match.append(has_match)

            neg_is_match = []
            for result in hard_neg_results:
                d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                    result, d_tokenizer, return_all_ids=True
                )
                d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
                d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
                has_match = False
                item_found = False
                for s, e in q_phrases:
                    if e - s == 1:
                        continue
                    # Check if is match
                    item_found = True
                    is_match = is_q_d_phrase_match(
                        q_toks=q_toks,
                        d_toks=d_toks,
                        q_start_idx=s,
                        q_end_idx=e,
                        d_max_token_idx=d_max_token_idx,
                    )
                    has_match = has_match or is_match
                if item_found:
                    neg_is_match.append(has_match)
            if pos_is_match and neg_is_match:
                # Check if the match is different between pos and neg
                only_pos_match_phrase += all(pos_is_match) and False in neg_is_match
                only_neg_match_phrase += all(neg_is_match) and True not in pos_is_match

        # Check q-d match for pos
        pos_is_match = []
        for result in pos_results:
            d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                result, d_tokenizer, return_all_ids=True
            )
            d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
            d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
            has_match = False
            item_found = False
            for s, e in q_word_indices:
                if e - s == 1:
                    continue
                # Check if is match
                item_found = True
                is_match = is_q_d_token_match(
                    q_toks=q_toks,
                    d_toks=d_toks,
                    q_start_idx=s,
                    q_end_idx=e,
                    d_max_token_idx=d_max_token_idx,
                    tag=inner_key,
                )
                has_match = has_match or is_match
            if item_found:
                pos_is_match.append(has_match)

        # Check q-d match for hard negs
        neg_is_match = []
        for result in hard_neg_results:
            d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                result, d_tokenizer, return_all_ids=True
            )
            d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
            d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
            has_match = False
            item_found = False
            for s, e in q_word_indices:
                if e - s == 1:
                    continue
                item_found = True
                # Check if is match
                is_match = is_q_d_token_match(
                    q_toks=q_toks,
                    d_toks=d_toks,
                    q_start_idx=s,
                    q_end_idx=e,
                    d_max_token_idx=d_max_token_idx,
                    tag=inner_key,
                )
                has_match = has_match or is_match
            if item_found:
                neg_is_match.append(has_match)
        if pos_is_match and neg_is_match:
            # Check if the match is different between pos and neg
            only_pos_match += all(pos_is_match) and False in neg_is_match
            only_neg_match += all(neg_is_match) and True not in pos_is_match
        stop = 1

    print(stat)
    print("Only pos match: ", only_pos_match)
    print("Only neg match: ", only_neg_match)
    print("Only pos match phrase: ", only_pos_match_phrase)
    print("Only neg match phrase: ", only_neg_match_phrase)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        # required=True,
        help="Path to the result file.",
        default="/root/ColBERT/debug/result.fever_baseline_nway32_q4_less_hard_lr2_distill.pkl",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        help="Dataset name.",
        default="fever",
    )
    parser.add_argument(
        "--q_max_len",
        type=int,
        default=64,
        help="Max length of the query.",
    )
    parser.add_argument(
        "--d_max_len",
        type=int,
        default=300,
        help="Max length of the document.",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="all",
        help="Type of evaluation.",
        choices=["all", "any"],
    )
    parser.add_argument(
        "--for_copy",
        action="store_true",
        help="Print the stats for copy.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    main(
        dataset_name=args.dataset,
        result_file_path=args.file_path,
        q_max_len=args.q_max_len,
        d_max_len=args.d_max_len,
        eval_type=args.eval_type,
        for_copy=args.for_copy,
    )

    logger.info(f"Done!")
