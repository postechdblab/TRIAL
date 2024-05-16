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


def remove_padding(scores: List[float]) -> List[float]:
    # Go reverse the list and split when meets non-zero for the first time
    for i in range(len(scores) - 1, -1, -1):
        if scores[i] != 0:
            return scores[: i + 1]


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


# Compute standard deviation
def std(scores: List[float]) -> float:
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return (
        (sum([(s - avg_score) ** 2 for s in scores]) / len(scores)) ** 0.5
        if scores
        else 0.0
    )


# Compute variance
def variance(scores: List[float]) -> float:
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return sum([(s - avg_score) ** 2 for s in scores]) / len(scores) if scores else 0.0


def avg(scores: List[float]) -> float:
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0


def print_stat(
    scores: List[float],
    name: str = None,
    total_tok_count: int = None,
    for_copy: bool = False,
    for_table: bool = False,
) -> Optional[Tuple]:
    max_score = max(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    std_score = (
        (sum([(s - avg_score) ** 2 for s in scores]) / len(scores)) ** 0.5
        if scores
        else 0.0
    )
    if total_tok_count is None or total_tok_count == 0:
        total_tok_count = 1
    total_percentage = round(len(scores) / total_tok_count * 100, 2)

    if for_table:
        return [
            str(round(item, 4))
            for item in (max_score, min_score, avg_score, std_score, len(scores))
        ]
    elif for_copy:
        logger.info(
            f"{max_score}, {min_score}, {avg_score}, {std_score}, {len(scores)}"
        )
    else:
        logger.info(
            f"Average {name} score: {avg_score} (max: {max_score} min: {min_score} std: {std_score}, total: {len(scores)} ({total_percentage}%))"
        )
    return None


def print_for_copy(scores_dict: Dict[str, Dict[str, List[float]]]) -> None:
    for outer_key in OUTER_KEYS:
        for inner_key in INNER_KEYS:
            for type_key in TYPE_KEYS:
                scores = scores_dict[type_key][f"{outer_key}_{inner_key}"]
                all_scores = scores_dict[type_key][f"{outer_key}_all"]
                print("")
                logger.info(
                    f"type_key: {type_key}, outer_key: {outer_key} inner_key: {inner_key}"
                )
                print_stat(
                    scores, name=None, total_tok_count=len(all_scores), for_copy=True
                )
    return None


def get_result_score(result) -> float:
    return sum(result.token_scores.max(axis=1))


def draw_table(
    scores_dict: Dict[str, Dict[str, List[float]]], dataset_name: str
) -> None:
    columns = ("MAX", "MIN", "AVERAGE", "STD", "# tokens")
    rows = (
        ["All"]
        + TYPE_KEYS
        + ["All: Pos. Doc."]
        + TYPE_KEYS
        + ["All: Neg. Doc."]
        + TYPE_KEYS
        + ["Incor: Pos. Doc."]
        + TYPE_KEYS
        + ["Incor: Neg. Doc."]
        + TYPE_KEYS
    )
    dummy_row = ["" for _ in range(len(columns))]
    # Create a new figure with a specific size
    plt.figure(figsize=(len(columns) * 2 * 1.3, len(rows) * 0.5))

    # Add a table at the bottom of the axes
    # 1
    cells = []
    cells += [dummy_row]
    for type_key in TYPE_KEYS:
        # all, all, type_key
        scores = scores_dict[type_key]["all_all"]
        values = print_stat(scores, name=None, for_table=True)
        cells.append(values)
    # 2
    cells += [dummy_row]
    for type_key in TYPE_KEYS:
        # correct, pos, type_key
        scores = scores_dict[type_key]["all_pos"]
        values = print_stat(scores, name=None, for_table=True)
        cells.append(values)
    # 3
    cells += [dummy_row]
    for type_key in TYPE_KEYS:
        # correct, neg, type_key
        scores = scores_dict[type_key]["all_neg"]
        values = print_stat(scores, name=None, for_table=True)
        cells.append(values)
    # 4
    cells += [dummy_row]
    for type_key in TYPE_KEYS:
        # incorrect, pos, type_key
        scores = scores_dict[type_key]["incorrect_pos"]
        values = print_stat(scores, name=None, for_table=True)
        cells.append(values)
    # 5
    cells += [dummy_row]
    for type_key in TYPE_KEYS:
        # incorrect, neg, type_key
        scores = scores_dict[type_key]["incorrect_neg"]
        values = print_stat(scores, name=None, for_table=True)
        cells.append(values)

    plt.table(cellText=cells, colLabels=columns, rowLabels=rows, loc="center")

    # Adjust layout to make room for the table:
    plt.axis("off")  # Hide axes

    file_name = f"{dataset_name}_stat.png"
    plt.savefig(file_name)
    plt.close()
    return None


def draw_table2(
    score_dict: Dict[str, Dict[str, Dict[str, List[float]]]], dataset_name: str
) -> None:
    columns = (
        ["Word (std)", "Prop. Noun (std)", "Noun (std)"]
        + ["Word (range)", "Prop. Noun (range)", "Noun (range)"]
        + ["Word (avg)", "Prop. Noun (avg)", "Noun (avg)"]
        + ["Word (cnt)", "Prop. Noun (cnt)", "Noun (cnt)"]
    )
    rows = (
        ["All"]
        + ["all", "pos", "neg"]
        + ["Correct"]
        + ["all", "pos", "neg"]
        + ["Incorrect"]
        + ["all", "pos", "neg"]
    )
    dummy_row = ["" for _ in range(len(columns))]
    # Create a new figure with a specific size
    plt.figure(figsize=(len(columns) * 2 * 1.3, len(rows) * 0.5))

    # Add a table at the bottom of the axes
    # 1
    cells = []
    for outer_key in OUTER_KEYS:
        cells += [dummy_row]
        for inner_key in INNER_KEYS:
            row = []
            for type_key in ["word", "prop_noun", "noun"]:
                scores = score_dict["std"][type_key][outer_key][inner_key]
                row.append(avg(scores))
            for type_key in ["word", "prop_noun", "noun"]:
                scores = score_dict["range"][type_key][outer_key][inner_key]
                row.append(avg(scores))
            for type_key in ["word", "prop_noun", "noun"]:
                scores = score_dict["avg"][type_key][outer_key][inner_key]
                row.append(avg(scores))
            for type_key in ["word", "prop_noun", "noun"]:
                scores = score_dict["std"][type_key]["all"][inner_key]
                row.append(len(scores))
            cells.append(row)
    plt.table(cellText=cells, colLabels=columns, rowLabels=rows, loc="center")

    # Adjust layout to make room for the table:
    plt.axis("off")  # Hide axes

    file_name = f"{dataset_name}_stat2.png"
    plt.savefig(file_name)
    plt.close()
    return None


def draw_table3(
    score_dict: Dict[str, Dict[str, Dict[str, List[float]]]], dataset_name: str
) -> None:
    columns = ["std", "diff"]
    rows = (
        ["All"]
        + ["all", "pos", "neg"]
        + ["Correct"]
        + ["all", "pos", "neg"]
        + ["Incorrect"]
        + ["all", "pos", "neg"]
    )
    dummy_row = ["" for _ in range(len(columns))]
    # Create a new figure with a specific size
    plt.figure(figsize=(len(columns) * 3 * 1.3, len(rows) * 0.5))

    # Add a table at the bottom of the axes
    # 1
    cells = []
    for outer_key in OUTER_KEYS:
        cells += [dummy_row]
        for inner_key in INNER_KEYS:
            row = []
            scores = score_dict["std"][outer_key][inner_key]
            row.append(avg(scores))
            scores = score_dict["diff"][outer_key][inner_key]
            row.append(avg(scores))
            cells.append(row)
    plt.table(cellText=cells, colLabels=columns, rowLabels=rows, loc="center")

    # Adjust layout to make room for the table:
    plt.axis("off")  # Hide axes

    file_name = f"{dataset_name}_stat3.png"
    plt.savefig(file_name)
    plt.close()
    return None


def aggregate_scores(
    scores_dict: Dict[str, Dict[str, List]],
    scores: List[float],
    indices: List[int],
    name: str,
    sub_name: str,
) -> None:
    # Append all results
    for idx in indices:
        scores_dict[name][sub_name].append(scores[idx])
    return None


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

    # Initialize the standard deviation for the scores
    multi_tok_max_min_range = {
        "word": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "prop_noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
    }
    multi_tok_std_dev = {
        "word": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "prop_noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
    }
    multi_tok_average = {
        "word": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "prop_noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "noun": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
    }
    multi_tok_matched = {
        "word_all_token": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
        "word_per_query": {
            "all": {"all": [], "pos": [], "neg": []},
            "correct": {"all": [], "pos": [], "neg": []},
            "incorrect": {"all": [], "pos": [], "neg": []},
        },
    }
    # Initialize total differences
    noun_prop_noun_differences_total = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    noun_stop_differences_total = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    noun_special_differences_total = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    noun_others_differences_total = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    prop_noun_stop_differences_total = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    prop_noun_special_differences_total = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    prop_noun_others_differences_total = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    # Initialize average differences
    noun_prop_noun_differences_avg = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    noun_stop_differences_avg = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    noun_special_differences_avg = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    noun_others_differences_avg = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    prop_noun_stop_differences_avg = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    prop_noun_special_differences_avg = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    prop_noun_others_differences_avg = {
        "c": {"p": [], "n": []},
        "i": {"p": [], "n": []},
    }
    num_of_noun_in_query = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    num_of_prop_noun_in_query = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    num_of_stop_in_query = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    num_of_special_in_query = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    num_of_others_in_query = {"c": {"p": [], "n": []}, "i": {"p": [], "n": []}}
    # Initialize the difference across document types
    pos_neg_token_diff_total = []
    pos_neg_token_diff_avg = []
    # For counting neg results without interesting tokens
    num_queries_with_neg_results_wo_interesting_tokens = 0
    num_neg_results_wo_interesting_tokens = 0
    num_negative_results = 0
    num_negative_results_when_incorrect = 0
    num_positive_results = 0
    num_positive_results_retrieved = 0
    is_correct_cnt = 0
    num_queries_with_pos_results_bad_effect = 0
    num_positive_results_bad_effect = 0
    # For evaluation on new ranking
    all_gold_reference: Dict[str, Dict[str, int]] = {}
    all_new_ranking: Dict[str, Dict[str, int]] = {}
    all_original_ranking: Dict[str, Dict[str, int]] = {}
    all_new_scores: Dict[str, Dict[str, List[float]]] = {}
    all_original_scores: Dict[str, Dict[str, List[float]]] = {}
    all_max_d_token_idx_std: Dict[str, Dict[str, float]] = {
        "all": {"all": [], "pos": [], "neg": []},
        "correct": {"all": [], "pos": [], "neg": []},
        "incorrect": {"all": [], "pos": [], "neg": []},
    }
    all_max_d_token_idx_diff: Dict[str, Dict[str, float]] = {
        "all": {"all": [], "pos": [], "neg": []},
        "correct": {"all": [], "pos": [], "neg": []},
        "incorrect": {"all": [], "pos": [], "neg": []},
    }
    all_is_found_for_query = []
    all_is_found_for_incorrect_query = []
    all_is_found_for_incorrect_query_and_pos_within_top_10 = []
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
        num_negative_results += len(neg_results)
        num_positive_results += len(pos_results)
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

        # Initialize
        is_found_cnt_within_query = []
        # Get token scores and document for positive results
        all_pos_q_tok_scores = []
        all_pos_d_max_toks = []
        all_pos_d_toks = []
        for result in pos_results:
            d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                result, d_tokenizer, return_all_ids=True
            )
            d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
            d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
            all_pos_d_max_toks.append(d_max_toks)
            all_pos_q_tok_scores.append(result.token_scores.max(axis=1))
            is_found_cnt_within_query.append([])
            all_pos_d_toks.append(d_toks)

        for inner_key, results in zip(
            ["pos", "neg", "all"], [pos_results, hard_neg_results, all_results]
        ):
            all_matched = {}
            all_matched["correct"] = None
            all_matched["incorrect"] = None

            for ri, result in enumerate(results):
                # d_tensor, d_mask = d_tokenizer.tensorize_packed([result.doc.text])
                pid = str(result.doc.id)
                # Get the max scores for each query tokens
                q_scores = result.token_scores.max(axis=1)
                # Create indices without padding
                non_pad_indices = [i for i, s in enumerate(q_scores) if s != 0.0]
                assert len(non_pad_indices) == len(
                    q_toks
                ), f"Length mismatch check tokenization: {len(non_pad_indices)} != {len(q_toks)}"
                for type_key in TYPE_KEYS:
                    # Set the outer key
                    outer_key = "correct" if is_correct else "incorrect"
                    # Create indices for the target token scores
                    target_indices = None
                    if type_key == "all":
                        target_indices = non_pad_indices
                    elif type_key == "noun":
                        target_indices = q_noun_indices
                    elif type_key == "prop_noun":
                        target_indices = q_prop_noun_indices
                    elif type_key == "stop":
                        target_indices = q_stop_indices
                    elif type_key == "special":
                        target_indices = q_special_indices
                    elif type_key == "others":
                        target_indices = q_others_indices
                    # Aggregate for all in outer key
                    aggregate_scores(
                        scores_dict=scores_dict,
                        scores=q_scores,
                        indices=target_indices,
                        name=type_key,
                        sub_name=f"all_{inner_key}",
                    )
                    # Aggregate for correct/incorrect in outer key
                    aggregate_scores(
                        scores_dict=scores_dict,
                        scores=q_scores,
                        indices=target_indices,
                        name=type_key,
                        sub_name=f"{outer_key}_{inner_key}",
                    )

                # Figure out the max score for each token
                if show_granularity_problem:
                    d_max_token_idx, d_max_token_id, d_ids = get_max_doc_tokens(
                        result, d_tokenizer, return_all_ids=True
                    )
                    d_max_toks = d_tokenizer.tok.convert_ids_to_tokens(d_max_token_id)
                    d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids)
                    d_max_token_idx_wo_pad = remove_padding(d_max_token_idx)[2:-1]
                    # Compute std
                    if d_max_token_idx_wo_pad:
                        d_max_token_idx_std = std(d_max_token_idx_wo_pad)
                        d_max_token_idx_diff = max(d_max_token_idx_wo_pad) - min(
                            d_max_token_idx_wo_pad
                        )
                        all_max_d_token_idx_std[outer_key][inner_key].append(
                            d_max_token_idx_std
                        )
                        all_max_d_token_idx_std["all"][inner_key].append(
                            d_max_token_idx_std
                        )
                        all_max_d_token_idx_diff[outer_key][inner_key].append(
                            d_max_token_idx_diff
                        )
                        all_max_d_token_idx_diff["all"][inner_key].append(
                            d_max_token_idx_diff
                        )

                    # Check contextualized token scores
                    if inner_key == "neg":
                        # Loop over query tokens and check if there exists case where:
                        #   1. the score is bigger for negative document
                        #   2. the max d token  is mismatch for positive and match for negative
                        # Check for all positive documents
                        for pi, (
                            pos_q_tok_scores,
                            pos_d_max_toks,
                            pos_d_toks,
                        ) in enumerate(
                            zip(
                                all_pos_q_tok_scores, all_pos_d_max_toks, all_pos_d_toks
                            )
                        ):
                            is_found = False
                            # Loop over all query tokens
                            neg_q_toks_scores = q_scores
                            neg_q_max_toks = d_max_toks
                            for qi, (q_tok, neg_q_score, neg_q_max_tok) in enumerate(
                                zip(q_toks, neg_q_toks_scores, neg_q_max_toks)
                            ):
                                pos_q_tok_score = pos_q_tok_scores[qi]
                                pos_d_max_tok = pos_d_max_toks[qi]
                                # Compare the score
                                if neg_q_score > pos_q_tok_score:
                                    # Compare the max token
                                    if (
                                        q_tok == neg_q_max_tok
                                        and q_tok != pos_d_max_tok
                                        and q_tok not in pos_d_toks
                                    ):
                                        is_found = True
                                        break
                                pass
                            is_found_cnt_within_query[pi].append(is_found)

                    # Add the score if not in the dict
                    if qid not in all_new_scores:
                        all_new_scores[qid] = {}
                        all_original_scores[qid] = {}
                    if pid not in all_new_scores[qid]:
                        original_token_scores = result.token_scores.max(axis=1)
                        all_new_scores[qid][pid] = copy.deepcopy(original_token_scores)
                        all_original_scores[qid][pid] = copy.deepcopy(
                            original_token_scores
                        )

                    # Calculate std for each word
                    for s, e in q_word_indices:
                        if e - s == 1:
                            continue
                        # if bool(set(q_toks[s:e]) & set(d_max_toks[s:e])):
                        scores = [q_scores[i] for i in range(s, e)]
                        std_score = std(scores)
                        multi_tok_max_min_range["word"][outer_key][inner_key].append(
                            max(scores) - min(scores)
                        )
                        multi_tok_std_dev["word"][outer_key][inner_key].append(
                            std_score
                        )
                        multi_tok_average["word"][outer_key][inner_key].append(
                            avg(scores)
                        )
                        multi_tok_max_min_range["word"]["all"][inner_key].append(
                            max(scores) - min(scores)
                        )
                        multi_tok_std_dev["word"]["all"][inner_key].append(std_score)
                        multi_tok_average["word"]["all"][inner_key].append(avg(scores))
                        # Check if is match
                        is_match = is_q_d_token_match(
                            q_toks=q_toks,
                            d_toks=d_toks,
                            q_start_idx=s,
                            q_end_idx=e,
                            d_max_token_idx=d_max_token_idx,
                            tag=inner_key,
                        )
                        all_matched[outer_key] = (
                            all_matched[outer_key] and is_match
                            if all_matched[outer_key] is not None
                            else is_match
                        )
                        multi_tok_matched["word_all_token"][outer_key][
                            inner_key
                        ].append(is_match)
                        # Change the token scores to 0 for the tokens if not matched
                        if not is_match:
                            if only_negative_for_rescoring:
                                if pid in neg_pids:
                                    # Update the token scores
                                    for i in range(s, e):
                                        all_new_scores[qid][pid][i] = 0.0
                            else:
                                # Update the token scores
                                for i in range(s, e):
                                    all_new_scores[qid][pid][i] = 0.0

                    # Calculate std for prop_noun
                    for s, e in q_phrases_prop:
                        if e - s == 1:
                            continue
                        scores = [q_scores[i] for i in range(s, e)]
                        std_score = std(scores)
                        multi_tok_max_min_range["prop_noun"][outer_key][
                            inner_key
                        ].append(max(scores) - min(scores))
                        multi_tok_std_dev["prop_noun"][outer_key][inner_key].append(
                            std_score
                        )
                        multi_tok_average["prop_noun"][outer_key][inner_key].append(
                            avg(scores)
                        )
                        multi_tok_max_min_range["prop_noun"]["all"][inner_key].append(
                            max(scores) - min(scores)
                        )
                        multi_tok_std_dev["prop_noun"]["all"][inner_key].append(
                            std_score
                        )
                        multi_tok_average["prop_noun"]["all"][inner_key].append(
                            avg(scores)
                        )

                    # Calculate std for noun
                    for s, e in q_phrases_noun:
                        if e - s == 1:
                            continue
                        scores = [q_scores[i] for i in range(s, e)]
                        std_score = std(scores)
                        multi_tok_max_min_range["noun"][outer_key][inner_key].append(
                            max(scores) - min(scores)
                        )
                        multi_tok_std_dev["noun"][outer_key][inner_key].append(
                            std_score
                        )
                        multi_tok_average["noun"][outer_key][inner_key].append(
                            avg(scores)
                        )
                        multi_tok_max_min_range["noun"]["all"][inner_key].append(
                            max(scores) - min(scores)
                        )
                        multi_tok_std_dev["noun"]["all"][inner_key].append(std_score)
                        multi_tok_average["noun"]["all"][inner_key].append(avg(scores))

                # Find the score difference b/w two types of tokens
                if show_token_type_difference and inner_key in ["pos", "neg"]:
                    # Get noun token scores
                    noun_token_scores = q_scores[q_noun_indices]
                    prop_noun_token_scores = q_scores[q_prop_noun_indices]
                    stop_token_scores = q_scores[q_stop_indices]
                    special_token_scores = q_scores[q_special_indices]
                    others_token_scores = q_scores[q_others_indices]
                    # Get labels
                    correctness_label = "c" if is_correct else "i"
                    document_label = "p" if inner_key == "pos" else "n"
                    # Count number of tokens in query
                    if len(noun_token_scores):
                        num_of_noun_in_query[correctness_label][document_label].append(
                            len(noun_token_scores)
                        )
                    if len(prop_noun_token_scores):
                        num_of_prop_noun_in_query[correctness_label][
                            document_label
                        ].append(len(prop_noun_token_scores))
                    if len(stop_token_scores):
                        num_of_stop_in_query[correctness_label][document_label].append(
                            len(stop_token_scores)
                        )
                    if len(special_token_scores):
                        num_of_special_in_query[correctness_label][
                            document_label
                        ].append(len(special_token_scores))
                    if len(others_token_scores):
                        num_of_others_in_query[correctness_label][
                            document_label
                        ].append(len(others_token_scores))
                    # 1. noun vs prop_noun
                    if len(noun_token_scores) and len(prop_noun_token_scores):
                        noun_prop_noun_differences_total[correctness_label][
                            document_label
                        ].append(sum(noun_token_scores) - sum(prop_noun_token_scores))
                        noun_prop_noun_differences_avg[correctness_label][
                            document_label
                        ].append(avg(noun_token_scores) - avg(prop_noun_token_scores))
                    # 2. noun vs stop
                    if len(noun_token_scores) and len(stop_token_scores):
                        noun_stop_differences_total[correctness_label][
                            document_label
                        ].append(sum(noun_token_scores) - sum(stop_token_scores))
                        noun_stop_differences_avg[correctness_label][
                            document_label
                        ].append(avg(noun_token_scores) - avg(stop_token_scores))
                    # 3. noun vs special
                    if len(noun_token_scores) and len(special_token_scores):
                        noun_special_differences_total[correctness_label][
                            document_label
                        ].append(sum(noun_token_scores) - sum(special_token_scores))
                        noun_special_differences_avg[correctness_label][
                            document_label
                        ].append(avg(noun_token_scores) - avg(special_token_scores))
                    # 4. noun vs others
                    if len(noun_token_scores) and len(others_token_scores):
                        noun_others_differences_total[correctness_label][
                            document_label
                        ].append(sum(noun_token_scores) - sum(others_token_scores))
                        noun_others_differences_avg[correctness_label][
                            document_label
                        ].append(avg(noun_token_scores) - avg(others_token_scores))
                    # 5. prop_noun vs stop
                    if len(prop_noun_token_scores) and len(stop_token_scores):
                        prop_noun_stop_differences_total[correctness_label][
                            document_label
                        ].append(sum(prop_noun_token_scores) - sum(stop_token_scores))
                        prop_noun_stop_differences_avg[correctness_label][
                            document_label
                        ].append(avg(prop_noun_token_scores) - avg(stop_token_scores))
                    # 6. prop_noun vs special
                    if len(prop_noun_token_scores) and len(special_token_scores):
                        prop_noun_special_differences_total[correctness_label][
                            document_label
                        ].append(
                            sum(prop_noun_token_scores) - sum(special_token_scores)
                        )
                        prop_noun_special_differences_avg[correctness_label][
                            document_label
                        ].append(
                            avg(prop_noun_token_scores) - avg(special_token_scores)
                        )
                    # 7. prop_noun vs others
                    if len(prop_noun_token_scores) and len(others_token_scores):
                        prop_noun_others_differences_total[correctness_label][
                            document_label
                        ].append(sum(prop_noun_token_scores) - sum(others_token_scores))
                        prop_noun_others_differences_avg[correctness_label][
                            document_label
                        ].append(avg(prop_noun_token_scores) - avg(others_token_scores))

            # Check whether any unmatched results are in the query
            for key in ["correct", "incorrect"]:
                if all_matched[key] is not None:
                    multi_tok_matched["word_per_query"][key][inner_key].append(
                        all_matched[key]
                    )

        # Find the best score
        handled_num = 0
        for item in is_found_cnt_within_query:
            tmp_handled_num = sum(item)
            if tmp_handled_num > handled_num:
                handled_num = tmp_handled_num
        if handled_num:
            all_is_found_for_query += [handled_num]
            if not is_correct:
                all_is_found_for_incorrect_query += [handled_num]
                if (len(hard_neg_results) - handled_num) < 10:
                    all_is_found_for_incorrect_query_and_pos_within_top_10 += [
                        handled_num
                    ]

        # Update the ranking with extracted scores
        tmp_results = pos_results + hard_neg_results
        for result in tmp_results:
            pid = str(result.doc.id)
            if qid not in all_original_ranking:
                all_original_ranking[qid] = {}
            all_original_ranking[qid][pid] = sum(all_original_scores[qid][pid])
            if qid not in all_new_ranking:
                all_new_ranking[qid] = {}
            all_new_ranking[qid][pid] = sum(all_new_scores[qid][pid])

        # Append gold
        if qid not in all_gold_reference:
            all_gold_reference[qid] = {}
        for i, pid in enumerate(gold_pids):
            all_gold_reference[qid][pid] = gold_p_scores[i]

        if compare_token_between_pos_neg:
            # type_key = "noun"
            # tareget_type_indices = q_noun_indices
            type_key = "prop_noun"
            tareget_type_indices = q_prop_noun_indices
            target_token_ids = [q_ids[i] for i in tareget_type_indices]
            # Check if any of the positive document is included in the top-k retrieved documents
            pos_pids = [str(item.doc.id) for item in pos_results]
            top_k_pids = [str(item.doc.id) for item in top_k_reusults]
            retrieved_pos_pids = [p for p in pos_pids if p in top_k_pids]
            retrieved_pos_indices = [
                i for i, p in enumerate(pos_pids) if p in top_k_pids
            ]
            num_positive_results_retrieved += len(retrieved_pos_pids)

            # Check if there exists any positive results that can have bad effect
            if len(retrieved_pos_pids) > 0 and len(target_token_ids) > 0:
                bad_pos_found = False
                for retrieved_pos_idx in retrieved_pos_indices:
                    pos_result = pos_results[retrieved_pos_idx]
                    pos_scores = pos_result.token_scores.max(axis=1)
                    interested_pos_scores = [
                        pos_scores[i] for i in tareget_type_indices
                    ]
                    pos_score = sum(pos_scores)
                    # Get negative results that are smaller than the positive result
                    target_neg_results = []
                    for neg_result in neg_results:
                        neg_score = get_result_score(neg_result)
                        if neg_score < pos_score:
                            target_neg_results.append(neg_result)
                    # Check the max scores for interested tokens in the target neg result
                    # max_scores_from_neg = [-1000 for _ in q_prop_noun_indices]
                    max_from_neg = -1000
                    for neg_result in target_neg_results:
                        neg_max_scores = neg_result.token_scores.max(axis=1)
                        neg_scores = [neg_max_scores[i] for i in tareget_type_indices]
                        neg_scores_sum = sum(neg_scores)
                        if neg_scores_sum > max_from_neg:
                            max_from_neg = neg_scores_sum
                        # for i in range(len(neg_scores)):
                        #     if neg_scores[i] > max_scores_from_neg[i]:
                        #         max_scores_from_neg[i] = neg_scores[i]
                    diff_score = sum(interested_pos_scores) - max_from_neg
                    # diff_score = sum([pos - neg for pos, neg in zip(interested_pos_scores, max_scores_from_neg)])
                    if diff_score < 0:
                        bad_pos_found = True
                        num_positive_results_bad_effect += 1
                if bad_pos_found:
                    num_queries_with_pos_results_bad_effect += 1

        if compare_token_between_pos_neg and not is_correct:
            # Compare the difference between types of tokens
            # Filter the negative results: use only those that includes all tokens for proper nouns
            type_key = "prop_noun"
            tareget_type_indices = q_prop_noun_indices
            # type_key = "noun"
            # tareget_type_indices = q_noun_indices
            target_token_ids = [q_ids[i] for i in tareget_type_indices]
            if len(target_token_ids) > 0:
                target_neg_result_found = False
                for neg_result in neg_results:
                    # Tokenize the document
                    d_ids, d_mask = d_tokenizer.tensorize(
                        [neg_result.doc.text], bsize=1
                    )[0][0]
                    d_ids = d_ids.squeeze(0).tolist()
                    # Get the max scores for the negative results
                    neg_max_scores = neg_result.token_scores.max(axis=1)
                    neg_scores = [neg_max_scores[i] for i in tareget_type_indices]

                    # Select the positive results to compare (This is because there are cases where the number of positive results is larger than 10)
                    if len(pos_results) > 10:
                        tmp_pos_scores = []
                        for pos_result in pos_results:
                            pos_max_scores = pos_result.token_scores.max(axis=1)
                            tmp_pos_scores.append(sum(pos_max_scores))
                        sorted_indices = np.argsort(tmp_pos_scores)[::-1][:10]
                        filtered_pos_results = [pos_results[i] for i in sorted_indices]
                    else:
                        filtered_pos_results = pos_results
                    use_min = True
                    if use_min:
                        # Find the minimum score from the positive results
                        # min_scores_from_pos = [1000 for _ in q_prop_noun_indices]
                        min_from_pos = 1000
                        assert (
                            len(pos_results) > 0
                        ), f"No positive results found for qid: {qid}"
                        for pos_result in filtered_pos_results:
                            pos_max_scores = pos_result.token_scores.max(axis=1)
                            pos_scores = [
                                pos_max_scores[i] for i in tareget_type_indices
                            ]
                            pos_scores_sum = sum(pos_scores)
                            if pos_scores_sum < min_from_pos:
                                min_from_pos = pos_scores_sum
                            # for i in range(len(pos_scores)):
                            #     if pos_scores[i] < min_scores_from_pos[i]:
                            #         min_scores_from_pos[i] = pos_scores[i]
                        # Compute diff using the minimum scores
                        # diff_score = sum([pos - neg for pos, neg in zip(min_scores_from_pos, neg_scores)])
                        diff_score = min_from_pos - sum(neg_scores)
                    else:
                        # Aggregate over all positive results
                        for pos_result in filtered_pos_results:
                            pos_max_scores = pos_result.token_scores.max(axis=1)
                            pos_scores = [
                                pos_max_scores[i] for i in tareget_type_indices
                            ]
                            diff_score = sum(
                                [
                                    pos_scores[i] - neg_scores[i]
                                    for i in range(len(pos_scores))
                                ]
                            )
                    if diff_score > 0:
                        target_neg_result_found = True
                        num_neg_results_wo_interesting_tokens += 1
                if target_neg_result_found:
                    num_queries_with_neg_results_wo_interesting_tokens += 1

        # Tokenize document
        if do_doc_tokens:
            for d_i, result in enumerate(pos_results):
                doc: Document = result.doc
                parsed_d = SpacyModel()([doc.text])
                d_ids, d_mask = d_tokenizer.tensorize([doc.text], bsize=1)[0][0]
                d_toks = d_tokenizer.tok.convert_ids_to_tokens(d_ids[0])
                d_phrases = get_phrase_indices(
                    d_ids,
                    d_mask,
                    d_tokenizer.tok,
                    [doc.text],
                    parsed_d,
                    bsize=1,
                    prop_noun_only=True,
                )[0][0]
                d_ids_inference, d_mask_inference = d_tokenizer.tensorize_packed(
                    [doc.text]
                )

                # Figure out the phrase indices
                raise NotImplementedError("Not implemented yet")

    # Compare the evaluation results with the original and new ranking
    new_eval_result = EvaluateRetrieval.evaluate(
        all_gold_reference, all_new_ranking, [1, 3, 5, 10, 50]
    )
    originanl_eval_result = EvaluateRetrieval.evaluate(
        all_gold_reference, all_original_ranking, [1, 3, 5, 10, 50]
    )

    # Write the result
    if only_negative_for_rescoring:
        suffix = "_only_neg.json"
    else:
        suffix = ".json"
    file_utils.write_json_file(
        new_eval_result, f"/root/ColBERT/{dataset_name}_eval_new{suffix}"
    )
    file_utils.write_json_file(
        originanl_eval_result, f"/root/ColBERT/{dataset_name}_eval_original{suffix}"
    )

    # Calculate average for all differences
    if show_token_type_difference:
        for type_key in [
            "noun_prop_noun",
            "noun_stop",
            "noun_special",
            "noun_others",
            "prop_noun_stop",
            "prop_noun_special",
            "prop_noun_others",
        ]:
            for correctness_label in ["c", "i"]:
                for document_label in ["p", "n"]:
                    if type_key == "noun_prop_noun":
                        total_diff = noun_prop_noun_differences_total[
                            correctness_label
                        ][document_label]
                        avg_diff = noun_prop_noun_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "noun_stop":
                        total_diff = noun_stop_differences_total[correctness_label][
                            document_label
                        ]
                        avg_diff = noun_stop_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "noun_special":
                        total_diff = noun_special_differences_total[correctness_label][
                            document_label
                        ]
                        avg_diff = noun_special_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "noun_others":
                        total_diff = noun_others_differences_total[correctness_label][
                            document_label
                        ]
                        avg_diff = noun_others_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "prop_noun_stop":
                        total_diff = prop_noun_stop_differences_total[
                            correctness_label
                        ][document_label]
                        avg_diff = prop_noun_stop_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "prop_noun_special":
                        total_diff = prop_noun_special_differences_total[
                            correctness_label
                        ][document_label]
                        avg_diff = prop_noun_special_differences_avg[correctness_label][
                            document_label
                        ]
                    elif type_key == "prop_noun_others":
                        total_diff = prop_noun_others_differences_total[
                            correctness_label
                        ][document_label]
                        avg_diff = prop_noun_others_differences_avg[correctness_label][
                            document_label
                        ]
                    assert len(total_diff) == len(
                        avg_diff
                    ), f"Length mismatch: {len(total_diff)} != {len(avg_diff)}"
                    print(f"\n{type_key}_{correctness_label}_{document_label}")
                    print(f"Num: {len(total_diff)}")
                    if type_key.startswith("noun"):
                        print(
                            f"Num of noun in query: {avg(num_of_noun_in_query[correctness_label][document_label])}"
                        )
                    elif type_key.startswith("prop_noun"):
                        print(
                            f"Num of prop_noun in query: {avg(num_of_prop_noun_in_query[correctness_label][document_label])}"
                        )
                    print(f"Average of total differences: {avg(total_diff)}")
                    print(f"Average of average differences: {avg(avg_diff)}")

    if compare_token_between_pos_neg:
        avg_diff = pos_neg_token_diff_avg
        total_diff = pos_neg_token_diff_total
        print(f"Total number of queries: {len(eval_data)}")
        print(f"Total number of incorrect queries: {len(eval_data) - is_correct_cnt}")
        print(f"\nTotal negative results: {num_negative_results}")
        print(
            f"Total negative results when incorrect: {num_negative_results_when_incorrect}"
        )
        print(
            f"Num of queries with neg results without interesting tokens: {num_queries_with_neg_results_wo_interesting_tokens}"
        )
        print(
            f"Num of neg results without interesting tokens: {num_neg_results_wo_interesting_tokens}"
        )
        print(f"\nTotal positive results: {num_positive_results}")
        print(f"Total positive results retrieved: {num_positive_results_retrieved}")
        print(
            f"Num of queries with pos results that have bad effect: {num_queries_with_pos_results_bad_effect}"
        )
        print(
            f"Num of pos results that have bad effect: {num_positive_results_bad_effect}"
        )

    # Save max d token idx std and diff

    draw_table3(
        score_dict={
            "std": all_max_d_token_idx_std,
            "diff": all_max_d_token_idx_diff,
        },
        dataset_name=dataset_name,
    )

    if show_granularity_problem:
        draw_table2(
            score_dict={
                "range": multi_tok_max_min_range,
                "std": multi_tok_std_dev,
                "avg": multi_tok_average,
            },
            dataset_name=dataset_name,
        )

    if for_copy:
        print_for_copy(scores_dict)
    else:
        draw_table(scores_dict=scores_dict, dataset_name=dataset_name)

    print(
        f"Number of queries that could be improved with better contextualization: {len(all_is_found_for_query)} (out of {len(eval_data)})"
    )
    print(
        f"Average number of queries that could be improved  with better contextualization: {avg(all_is_found_for_query)}"
    )
    print(
        f"Number of incorrect queries that could be improved with better contextualization: {len(all_is_found_for_incorrect_query)} (out of {len(eval_data) - is_correct_cnt})"
    )
    print(
        f"Average number of incorrect queries that could be improved  with better contextualization: {avg(all_is_found_for_incorrect_query)}"
    )
    print(
        f"Number of incorrect queries that could be improved with better contextualization and pos within top 10: {len(all_is_found_for_incorrect_query_and_pos_within_top_10)} (out of {len(eval_data) - is_correct_cnt})"
    )
    print(
        f"Average number of incorrect queries that could be improved  with better contextualization and pos within top 10: {avg(all_is_found_for_incorrect_query_and_pos_within_top_10)}"
    )

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        # required=True,
        help="Path to the result file.",
        default="/root/ColBERT/debug/result.nq_baseline_nway32_q4_less_hard_lr2_distill.pkl",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # required=True,
        help="Dataset name.",
        default="nq",
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
