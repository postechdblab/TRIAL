import logging
import math
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.dataset.utils import read_compressed
from eagle.phrase.extraction2 import PhraseExtractor2 as PhraseExtractor
from eagle.tokenizer import Tokenizers

# from eagle.phrase.extraction import PhraseExtractor


logger = logging.getLogger("PhraseExtraction")

CHUNK_SIZE = 1000


def get_file_name(prefix: str) -> str:
    return f"phrase_indices.{prefix}.pkl"


def filter_file_names(
    file_names: List[str], total: int, prefix: str = None
) -> List[str]:
    file_names = [f for f in file_names if ".pkl." in f]
    filtered_files = []
    for file in file_names:
        try:
            num = int(file.split(".")[-1])
            if num >= total:
                continue
        except:
            continue
        if prefix is not None and prefix in file:
            filtered_files.append(file)
    assert (
        len(filtered_files) == total
    ), f"Total number of files is {len(filtered_files)} but expected {total}"
    return filtered_files


def analyze(text: str, tokenizer, phrase_indices_by_token: List[Tuple[int]]) -> None:
    print("Text:", text)
    tokens = tokenizer.tokenize(text)["input_ids"][0]
    tokens = tokenizer.tokenizer.convert_ids_to_tokens(tokens)
    print("Nouns:", [tokens[s:e] for s, e in phrase_indices_by_token])
    input("Is OK?")
    return None


def extract(
    cfg: DictConfig,
    dataset_name: str,
    split_i: int,
    total: int,
    prefix: str,
) -> None:
    logger.info(f"I: {split_i}, Total: {total}")
    dir_path = os.path.join("/root/EAGLE/data/", dataset_name)
    if prefix == "query":
        filename = "queries.jsonl"
    elif prefix == "doc":
        filename = "corpus.jsonl"
    else:
        raise ValueError(f"Invalid prefix: {prefix}")
    dataset_path = os.path.join(dir_path, filename)

    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    tokenizer = tokenizers.q_tokenizer if prefix == "query" else tokenizers.d_tokenizer
    SEP_TOKEN = tokenizer.tokenizer.special_tokens_map["sep_token"]
    extractor = PhraseExtractor(tokenizer=tokenizer)

    # Load tokenized text
    tokenized_path = os.path.join(
        dir_path, f"{filename}.{tokenizers.model_name}-tok.cache"
    )
    logger.info(f"Loading the tokenizered data from {tokenized_path}")
    tokenized_data = read_compressed(tokenized_path)

    # Get the file path
    file_name = get_file_name(prefix=prefix)
    if split_i == 0 and total == 1:
        is_split = False
    else:
        is_split = True
        file_name = f"{file_name}.{split_i}"
    output_file_path = os.path.join(dir_path, file_name)

    # Load the corpus
    logger.info(f"Loading the text data from {dataset_path}")
    dataset: List = file_utils.read_json_file(dataset_path, auto_detect_extension=True)
    logger.info(f"Loaded {len(dataset)} text.")

    # Divide the dataset into chunks and extract phrases
    chunks: List = list_utils.divide_into_chunks(dataset, num_chunks=total)
    target_chunk = chunks[split_i]
    logger.info(f"Target chunk size: {len(target_chunk)}")
    mini_chunks: List = list_utils.chunks(target_chunk, CHUNK_SIZE)

    all_results: Dict[int, List[List[Tuple[int]]]] = {}
    for ci, chunk in enumerate(
        tqdm.tqdm(mini_chunks, total=math.ceil(len(target_chunk) / CHUNK_SIZE))
    ):
        ids = [str(item["_id"]) for item in chunk]
        # TODO: First, perform phrase extraction for all sentences and then combine into query/document
        # TODO: Separate the extractor logic: 1) extract phrase indices by character 2) convert to token-level indices

        sents = []
        sent_lens = []
        tok_ids_in_sent = []
        SEP_TOKEN = tokenizer.tokenizer.special_tokens_map["sep_token"]
        SEP_TOKEN_ID = tokenizer.tokenizer.vocab[SEP_TOKEN]
        for sent in chunk:
            _id = str(sent["_id"])
            sents.extend(sent["text"])
            sent_lens.append(len(sent["text"]))
            tok_ids = tokenized_data[_id]
            # Split the token ids into sentences. sep_token is used to separate sentences
            tmp_tok_ids = []
            tmp_tok_ids_in_sent = []
            for tok_id in tok_ids:
                if tok_id == SEP_TOKEN_ID:
                    tmp_tok_ids_in_sent.append(tmp_tok_ids)
                    tmp_tok_ids = []
                else:
                    tmp_tok_ids.append(tok_id)
            # Append the last sentence
            if len(tmp_tok_ids):
                tmp_tok_ids_in_sent.append(tmp_tok_ids)
            tok_ids_in_sent.extend(tmp_tok_ids_in_sent)
            assert len(sent["text"]) == len(tmp_tok_ids_in_sent), f"Length of text: {len(sent['text'])} != Length of tok_ids_in_sent: {len(tmp_tok_ids_in_sent)}"
        assert len(sents) == len(tok_ids_in_sent), f"Length of sents: {len(sents)} != Length of tok_ids_in_sent: {len(tok_ids_in_sent)}"

        texts = [f" {tokenizer.tokenizer.special_tokens_map["sep_token"]} ".join(item["text"]) for item in chunk]
        tok_ids = [tokenized_data[_id] for _id in ids]

        results = extractor(
            texts,
            max_tok_len=tokenizer.cfg.max_len,
            tok_ids=tok_ids,
        )
        all_results[ci] = results

    # Convert format
    if not is_split:
        all_results_tmp = []
        for key in sorted(all_results.keys()):
            all_results_tmp.extend(all_results[key])
        all_results = all_results_tmp

    file_utils.write_pickle_file(all_results, output_file_path)


def merge(
    dataset_path: str, output_dir_path: str, total: int, prefix: str
) -> None:
    output_file_path = os.path.join(
        output_dir_path, get_file_name(prefix=prefix)
    )
    # List all the files in the directory
    file_names = sorted(os.listdir(output_dir_path))
    file_names = filter_file_names(file_names, total, prefix=prefix)
    all_dict: Dict = {}
    logger.info(f"Reading total number of cache shards: {len(file_names)}")
    for file_name in tqdm.tqdm(file_names):
        shard_num = int(file_name.split(".")[-1])
        file_path = os.path.join(output_dir_path, file_name)
        cached_dict = file_utils.read_pickle_file(file_path)
        # Aggregate into a single list
        all_values: List[List[Tuple[int, int]]] = []

        max_key = max(cached_dict.keys())
        for i in range(max_key + 1):
            if i in cached_dict:
                all_values.extend(cached_dict[i])
            else:
                all_values.extend([] for _ in range(CHUNK_SIZE))
        # Add to the dictionary
        all_dict[shard_num] = all_values

    # Load corpus
    logger.info(f"Loading the dataset from {dataset_path}")
    dataset: List = file_utils.read_jsonl_file(dataset_path)

    # Flatten list
    all_values: List[List[Tuple[int, int]]] = []
    for i in range(total):
        all_values.extend(all_dict[i])

    # Perform further processing for empty values
    assert len(all_values) == len(
        dataset
    ), f"Length of all_values: {len(all_values)} != Length of dataset: {len(dataset)}"

    # Save the dictionary
    logger.info(
        f"Saving the dictionary of length {len(all_values)} to {output_file_path}"
    )
    file_utils.write_pickle_file(all_values, output_file_path)
    return None

@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Args:
        - op: merge, filter, extract
        - target_data: query, document
        - i (optional): split index
        - total: total number of splits
    """
    if cfg.target_data == "query":
        prefix = "query"
        dataset_path = cfg.dataset.query_file
    elif cfg.target_data == "document":
        prefix = "doc"
        dataset_path = cfg.dataset.corpus_file
    else:
        raise ValueError(f"Invalid type: {cfg.type}")

    # Get output directory
    output_dir_path = os.path.join("/root/EAGLE/data/", cfg.dataset.name)
    dataset_path = os.path.join(output_dir_path, dataset_path)

    if cfg.op == "merge":
        merge(
            dataset_path=dataset_path,
            output_dir_path=output_dir_path,
            total=cfg.total,
            prefix=prefix,
        )
    elif cfg.op == "extract":
        extract(
            cfg=cfg,
            dataset_name=cfg.dataset.name,
            split_i=cfg.i,
            total=cfg.total,
            prefix=prefix,
        )
    else:
        raise ValueError(f"Invalid operation: {cfg.op}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to extract phrase indices!",
        error_msg=f"Falied to extract phrase indices!",
    ):
        main()
    logger.info(f"Done!")
