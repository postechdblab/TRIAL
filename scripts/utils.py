import os
from typing import *

import hkkang_utils.file as file_utils
import torch
from omegaconf import open_dict

BEIR_DATASET_NAMES = [
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
    # "cqadupstack",
    # "bioasq",
    # "signal1m",
    # "trec-news",
    # "robust04",
]
LOTTE_DATASET_NAMES = [
    "lifestyle-forum",
    "lifestyle-search",
    "pooled-forum",
    "pooled-search",
    "recreation-forum",
    "recreation-search",
    "science-forum",
    "science-search",
    "technology-forum",
    "technology-search",
    "writing-forum",
    "writing-search",
]


def join_word(tokens: List[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end]).replace(" ##", "").replace("[PAD]", "")


def check_argument(
    dic: Dict,
    name: str,
    arg_type: Type,
    choices: List[Any] = None,
    is_requried: bool = False,
    help: str = None,
) -> bool:
    with open_dict(dic):
        # Check if the argument is required
        if is_requried and name not in dic:
            raise ValueError(f"{name} is required!.({help})")
        # Check argument type and choices
        if name in dic:
            if not isinstance(dic[name], arg_type):
                raise ValueError(f"{name} should be {arg_type}. ({help})")
            if choices is not None and dic[name] not in choices:
                raise ValueError(f"{name} should be in {choices}. ({help})")
        # Set default value for boolean args
        if name not in dic and arg_type == bool:
            dic[name] = False
    return True


def validate_dataset_name(dataset_name: str) -> bool:
    assert dataset_name in BEIR_DATASET_NAMES, f"Invalid dataset name: {dataset_name}"


def validate_model_name(checkpoint_dir, model_name: str) -> bool:
    assert os.path.exists(
        os.path.join(checkpoint_dir, model_name)
    ), f"Invalid model name: {model_name}"


# def load_tokenizer(
#     is_for_query: bool = True, model_name: Optional[str] = None
# ) -> Union[DocTokenizer, QueryTokenizer]:
#     if is_for_query:
#         tokenizer = QueryTokenizer(ColBERTConfig(checkpoint=model_name))
#     else:
#         tokenizer = DocTokenizer(ColBERTConfig(checkpoint=model_name))
#     return tokenizer


def read_qrels(qrels_path: str) -> List[str]:
    """Return a list of (query id, doc id) tuples.
    The doc id is the positive doc id for the query id."""
    qrels = file_utils.read_csv_file(
        qrels_path, delimiter="\t", first_row_as_header=False
    )
    tmp = []
    for item in qrels:
        if len(item) == 4:
            assert len(item) == 4, f"Invalid qrels item: {item}"
            assert item[1] == "0" and item[3] == "1", f"Invalid qrels item: {item}"
            tmp.append((item[0], item[2]))
        elif len(item) == 3:
            # Support beir format
            if item[0] == "query-id":
                continue
            assert len(item) == 3, f"Invalid qrels item: {item}"
            assert item[2] == "1", f"Invalid qrels item: {item}"
            tmp.append((item[0], item[1]))
    return tmp


def read_queries(queries_path: str) -> Dict[str, str]:
    """Return a dict of query id to query text."""
    queries = file_utils.read_csv_file(
        queries_path, delimiter="\t", first_row_as_header=False
    )
    tmp = {}
    for item in queries:
        assert len(item) == 2, f"Invalid queries item: {item}"
        assert item[0] not in tmp, f"Duplicate query id: {item[0]}"
        tmp[item[0]] = item[1]
    return tmp


def min_noun_scores(
    scores: torch.Tensor,
    query_tok_ids: List[int],
    noun_tok_id: List[int],
    do_times: bool = True,
) -> torch.Tensor:
    """Combine the noun word scores into a single score by applying min.
    :param scores: Shape (num_query_tok)
    :rtype: torch.Tensor
    """
    for q_idx in range(len(query_tok_ids)):
        if q_idx + len(noun_tok_id) < len(query_tok_ids):
            if query_tok_ids[q_idx : q_idx + len(noun_tok_id)] == noun_tok_id:
                min_score = torch.min(scores[q_idx : q_idx + len(noun_tok_id)], dim=0)[
                    0
                ]
                scores[q_idx : q_idx + len(noun_tok_id)] = min_score
    return scores


def get_noun_score(
    scores: torch.Tensor,
    query_tok_ids: List[int],
    noun_tok_id: List[int],
    do_times: bool = True,
) -> Union[torch.Tensor, None]:
    """Combine the noun word scores into a single score by applying min.
    :param scores: Shape (num_query_tok)
    :rtype: torch.Tensor
    """
    for q_idx in range(len(query_tok_ids)):
        if q_idx + len(noun_tok_id) < len(query_tok_ids):
            if query_tok_ids[q_idx : q_idx + len(noun_tok_id)] == noun_tok_id:
                return torch.min(scores[q_idx : q_idx + len(noun_tok_id)], dim=0)[0]
    return None


def weighted_scores(
    scores: torch.Tensor,
    query_tok_ids: List[int],
    target_tok_ids: List[int],
    ratio: float = 1.0,
) -> torch.Tensor:
    """Apply weighted to the target token ids.
    :param scores: Shape (num_query_tok)
    :rtype: torch.Tensor
    """
    for q_idx in range(len(query_tok_ids)):
        if q_idx + len(target_tok_ids) < len(query_tok_ids):
            if query_tok_ids[q_idx : q_idx + len(target_tok_ids)] == target_tok_ids:
                scores[q_idx : q_idx + len(target_tok_ids)] *= ratio
    return scores


def combine_tokenized_words(toks: List[str]) -> str:
    new_str = ""
    for tok in toks:
        if tok.startswith("##"):
            new_str += tok[2:]
        else:
            new_str += " " + tok
    return new_str.strip()


def average_noun_scores(
    input_values: torch.Tensor,
    input_ids: torch.Tensor,
    noun_tok_ids_list: List[List[int]],
) -> torch.Tensor:
    """Combine the noun word scores into a single score by applying min.
    :param scores: Shape (num_query_tok)
    :rtype: torch.Tensor
    """
    input_ids = input_ids.tolist()
    for tok_ids in noun_tok_ids_list:
        for i_idx in range(len(input_ids) - len(tok_ids) + 1):
            if input_ids[i_idx : i_idx + len(tok_ids)] == tok_ids:
                input_values[i_idx : i_idx + len(tok_ids)] = torch.mean(
                    input_values[i_idx : i_idx + len(tok_ids)], dim=0
                )
                break
    return input_values


# Function to print tokens and indices with appropriate formatting
def pretty_print_tokens_with_their_indices(
    decoded_tokens: List[str], max_tokens_per_line: int = 20
) -> None:
    for start in range(0, len(decoded_tokens), max_tokens_per_line):
        # Get the tokens for the current line (max 20 tokens)
        tokens_chunk = decoded_tokens[start : start + max_tokens_per_line]

        # Print tokens in a single line with two spaces between each token
        tokens_line = "  ".join(tokens_chunk)
        print(tokens_line)

        # Calculate the position of each index and print the indices centered below the tokens
        indices_line = ""
        for idx, token in enumerate(tokens_chunk, start=start):
            token_length = len(token)
            # Center the index under the token by calculating the appropriate padding
            padding = (token_length - len(str(idx))) // 2
            indices_line += f"{' ' * padding}{idx}{' ' * (token_length - padding - len(str(idx)))}  "  # Two spaces after each token

        print(indices_line)
