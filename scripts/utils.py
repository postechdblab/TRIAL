import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import torch
from omegaconf import open_dict
from torch.nn.utils.rnn import pad_sequence

from eagle.dataset.utils import get_att_mask, get_mask
from eagle.model.batch.utils import (
    convert_range_to_scatter,
    cut_off_phrase_ranges_by_max_len,
)
from eagle.phrase import PhraseExtractor
from eagle.phrase.utils import (
    combined_phrase_ranges_into_one_sentence,
    fix_bad_index_ranges,
)
from eagle.tokenization.sentencizer import Sentencizer
from eagle.tokenization.tokenizer import Tokenizer
from eagle.tokenization.utils import combine_splitted_tok_ids

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

logger = logging.getLogger("Utils")


def format_preprocessed_data_as_batch(
    preprocessed_query: Dict[str, Any],
    preprocessed_document: Dict[str, Any],
    model_device: str = None,
) -> Dict[str, Any]:
    # Create batch input
    preprocessed_batch = {
        "q_tok_ids": preprocessed_query["tok_ids"],
        "q_tok_att_mask": preprocessed_query["tok_att_mask"],
        "q_tok_mask": preprocessed_query["tok_mask"],
        "doc_tok_ids": preprocessed_document["tok_ids"].unsqueeze(0),
        "doc_tok_att_mask": preprocessed_document["tok_att_mask"].unsqueeze(0),
        "doc_tok_mask": preprocessed_document["tok_mask"].unsqueeze(0),
        "labels": None,
        "distillation_scores": None,
        "pos_doc_ids": None,
        "is_analyze": True,
        "q_sent_start_indices": [preprocessed_query["sent_start_indices"]],
        "doc_sent_start_indices": [preprocessed_document["sent_start_indices"]],
        "q_phrase_scatter_indices": (
            None
            if preprocessed_query["phrase_scatter_indices"] is None
            else preprocessed_query["phrase_scatter_indices"]
        ),
        "doc_phrase_scatter_indices": (
            None
            if preprocessed_document["phrase_scatter_indices"] is None
            else preprocessed_document["phrase_scatter_indices"]
        ),
        "q_phrase_mask": (
            None
            if preprocessed_query["phrase_mask"] is None
            else preprocessed_query["phrase_mask"]
        ),
        "doc_phrase_mask": (
            None
            if preprocessed_document["phrase_mask"] is None
            else preprocessed_document["phrase_mask"].unsqueeze(0)
        ),
        "q_sent_mask": (
            None
            if preprocessed_query["sent_mask"] is None
            else preprocessed_query["sent_mask"]
        ),
        "doc_sent_mask": (
            None
            if preprocessed_document["sent_mask"] is None
            else preprocessed_document["sent_mask"].unsqueeze(0)
        ),
    }
    if model_device is not None:
        # Move the tensors to the device same as the model
        preprocessed_batch = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed_batch.items()
        }

    return preprocessed_batch


def preprocess(
    text_batch: List[str], tokenizer: Tokenizer, extract_phrase: bool = True
) -> Any:
    # Split the text into sentences
    sentences_list: List[List[str]] = Sentencizer()(text_batch)
    sent_num_list: List[int] = [len(item) for item in sentences_list]
    flatten_sentences: List[str] = [
        item for sublist in sentences_list for item in sublist
    ]
    # Tokenize the text
    tokenized_sentences: List[List[int]] = tokenizer(flatten_sentences)["input_ids"]
    # Extract the phrases
    if extract_phrase:
        extractor = PhraseExtractor(tokenizer=tokenizer)
        # Extract phrases and combine them as a single sentence
        all_phrase_ranges: List[torch.Tensor] = []
        all_scatter_indices: List[torch.Tensor] = []
        cnt = 0
        for sent_num in sent_num_list:
            sentences = flatten_sentences[cnt : cnt + sent_num]
            target_tokenized_sentences = tokenized_sentences[cnt : cnt + sent_num]
            cnt += sent_num
            # Process
            phrase_ranges_per_sent: List[List[Tuple[int, int]]] = extractor(
                text_or_texts=sentences,
                tok_ids_or_tok_ids_list=target_tokenized_sentences,
                to_token_indices=True,
            )
            phrase_ranges = combined_phrase_ranges_into_one_sentence(
                [fix_bad_index_ranges(item) for item in phrase_ranges_per_sent]
            )
            phrase_ranges = cut_off_phrase_ranges_by_max_len(
                phrase_ranges, tokenizer.cfg.max_len
            )
            # Get scatter indices for phrases
            scatter_indices: List[int] = convert_range_to_scatter(phrase_ranges)
            # # Cut off phrase scatter indices if it exceeds the maximum length
            # scatter_indices = tokenizer.cutoff_by_max_len(
            #     scatter_indices, maintain_special_tokens=False
            # )
            scatter_indices = torch.tensor(scatter_indices, dtype=torch.long)
            all_scatter_indices.append(scatter_indices)
            all_phrase_ranges.append(phrase_ranges)
        # To Batch
        max_len = max(max(item) for item in all_scatter_indices)
        all_scatter_indices = pad_sequence(
            all_scatter_indices, batch_first=True, padding_value=max_len
        )

    # Preprocess as the model input
    # Combine the splitted sentences
    cnt = 0
    all_tok_ids = []
    all_sent_start_indices = []
    for sent_num in sent_num_list:
        selected_tokenized_sentences = tokenized_sentences[cnt : cnt + sent_num]
        tok_ids, sent_start_indices = combine_splitted_tok_ids(
            selected_tokenized_sentences
        )
        # Cut-off by max length
        tok_ids = tokenizer.cutoff_by_max_len(tok_ids)
        all_tok_ids.append(tok_ids)
        # Cut-off by max length
        all_sent_start_indices.append(
            [item for item in sent_start_indices if item <= tokenizer.cfg.max_len]
        )
        cnt += sent_num
    # Convert list to tensor
    tok_ids_tensor_list = [torch.tensor(item) for item in all_tok_ids]
    tok_ids_tensor = pad_sequence(
        tok_ids_tensor_list, batch_first=True, padding_value=0
    )
    # Create token mask
    tok_mask = get_mask(tok_ids_tensor, skip_ids=tokenizer.skip_tok_ids)
    tok_att_mask = get_att_mask(tok_ids_tensor, skip_ids=[0])
    if extract_phrase:
        phrase_masks = []
        for phrase_ranges in all_phrase_ranges:
            phrase_mask = torch.zeros(len(phrase_ranges), dtype=torch.bool).float()
            phrase_masks.append(phrase_mask)
        phrase_mask = pad_sequence(phrase_masks, batch_first=True, padding_value=1)
        sent_masks = []
        for sent_start_indices in all_sent_start_indices:
            sent_mask = torch.zeros(len(sent_start_indices), dtype=torch.bool).float()
            sent_masks.append(sent_mask)
        sent_mask = pad_sequence(sent_masks, batch_first=True, padding_value=1)
    else:
        phrase_mask = None
        sent_mask = None
        scatter_indices = None
        phrase_ranges = None

    return {
        "tok_ids": tok_ids_tensor,
        "tok_att_mask": tok_att_mask,
        "tok_mask": tok_mask,
        "sent_start_indices": all_sent_start_indices,
        "phrase_scatter_indices": all_scatter_indices,
        "phrase_mask": phrase_mask,
        "sent_mask": sent_mask,
        "phrase_ranges": all_phrase_ranges,
    }


def join_word(tokens: List[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end]).replace(" ##", "").replace("[PAD]", "")


def remove_model_prefix_key_from_saved_dict(ckpt_path: str) -> None:
    logger.info(f"Loding the checkpoint from {ckpt_path}")
    tmp = torch.load(ckpt_path, weights_only=True)
    logger.info(f"Removing the prefix from the model state_dict")
    tmp["state_dict"] = {
        k.replace("._orig_mod.", "."): v for k, v in tmp["state_dict"].items()
    }
    logger.info(f"Saving the modified checkpoint to {ckpt_path}")
    torch.save(tmp, ckpt_path)
    return None


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
        tokens_chunk = [token for token in tokens_chunk if token != "[PAD]"]

        # Print tokens in a single line with two spaces between each token
        tokens_line = "  ".join(tokens_chunk)
        print(tokens_line)

        # Calculate the position of each index and print the indices centered below the tokens
        indices_line = ""
        for idx, token in enumerate(tokens_chunk, start=start):
            if token == "[PAD]":
                continue
            token_length = len(token)
            # Center the index under the token by calculating the appropriate padding
            padding = (token_length - len(str(idx))) // 2
            indices_line += f"{' ' * padding}{idx}{' ' * (token_length - padding - len(str(idx)))}  "  # Two spaces after each token

        print(indices_line)
