import logging
import math
import os
from typing import *

import hkkang_utils.concurrent as concurrent_utils
import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.dataset.utils import read_compressed
from eagle.phrase.extraction2 import PhraseExtractor2 as PhraseExtractor
from eagle.tokenizer import Tokenizers

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
    partial_processor = concurrent_utils.PartialProcessor(
        total_proc_n=total, current_proc_n=split_i
    )
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
    dataset_chunk: List = partial_processor.get_partial_data(dataset)
    tokenized_data_chunk: List = partial_processor.get_partial_data(tokenized_data)

    logger.info(f"Target chunk size: {len(dataset_chunk)}")
    mini_dataset_chunks: Generator = list_utils.chunks(dataset_chunk, CHUNK_SIZE)
    mini_tokenized_data_chunks: List = list_utils.chunks(
        tokenized_data_chunk, CHUNK_SIZE
    )

    logger.info(f"Begin to extract phrases from {len(dataset_chunk)} texts")
    all_results: Dict[int, List[List[Tuple[int]]]] = {}
    for ci, (d_chunk, t_chunk) in enumerate(
        tqdm.tqdm(
            zip(mini_dataset_chunks, mini_tokenized_data_chunks, strict=True),
            total=math.ceil(len(dataset_chunk) / CHUNK_SIZE),
        )
    ):
        # TODO: First, perform phrase extraction for all sentences and then combine into query/document
        # TODO: Separate the extractor logic: 1) extract phrase indices by character 2) convert to token-level indices

        sents = []
        sent_lens = []
        tok_ids_in_sent = []
        for sent, sents_tok_ids in zip(d_chunk, t_chunk, strict=True):
            sents.extend(sent["text"])
            sent_lens.append(len(sent["text"]))
            tok_ids_in_sent.extend(sents_tok_ids)
            # Split the token ids into sentences. sep_token is used to separate sentences
            assert len(sent["text"]) == len(
                sents_tok_ids
            ), f"Length of text: {len(sent['text'])} != Length of tok_ids_in_sent: {len(sents_tok_ids)}"
        assert len(sents) == len(
            tok_ids_in_sent
        ), f"Length of sents: {len(sents)} != Length of tok_ids_in_sent: {len(tok_ids_in_sent)}"

        results = extractor(
            texts=sents,
            tok_ids=tok_ids_in_sent,
            to_token_indices=True,
        )
        all_results[ci] = results

    # Convert format
    if not is_split:
        all_results_tmp = []
        for key in sorted(all_results.keys()):
            all_results_tmp.extend(all_results[key])
        all_results = all_results_tmp

    logger.info(f"Saving the results to {output_file_path}")
    file_utils.write_pickle_file(all_results, output_file_path)


def merge(dataset_path: str, output_dir_path: str, total: int, prefix: str) -> None:
    output_file_path = os.path.join(output_dir_path, get_file_name(prefix=prefix))
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
