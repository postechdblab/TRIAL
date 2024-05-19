import logging
import math
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import hydra
import tqdm

from eagle.dataset.utils import extract_word_range_with_multi_tokens
from eagle.phrase.extraction import PhraseExtractor
from eagle.tokenizer import NewTokenizer

logger = logging.getLogger("PhraseExtraction")

CHUNK_SIZE = 1000


def get_file_name(prefix: str, index_type: str) -> str:
    return f"{index_type}_indices.{prefix}.pkl"


def fileter_file_names(file_names: List[str], total: int) -> List[str]:
    file_names = [f for f in file_names if ".pkl." in f]
    filtered_files = []
    for file in file_names:
        try:
            num = int(file.split(".")[-1])
            if num >= total:
                continue
        except:
            continue
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
    dir_path: str,
    dataset_path: str,
    tokenizer_cfg: Dict,
    split_i: int,
    total: int,
    index_type: str,
    prefix: str,
) -> None:
    # Configs
    logger.info(f"I: {split_i}, Total: {total}")
    tokenizer = NewTokenizer(tokenizer_cfg)
    extractor = PhraseExtractor(tokenizer=tokenizer)

    # Get the file path
    file_name = get_file_name(prefix=prefix, index_type=index_type)
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
        texts = [item["text"] for item in chunk]
        if index_type == "word":
            results = tokenizer(texts)["input_ids"]
            # Convert to token text
            toks_list = [
                tokenizer.tokenizer.convert_ids_to_tokens(tok_ids)
                for tok_ids in results
            ]
            results = [extract_word_range_with_multi_tokens(toks) for toks in toks_list]
        else:
            results = extractor(texts, max_len=tokenizer_cfg.max_len)
        all_results[ci] = results

    # Convert format
    if not is_split:
        all_results_tmp = []
        for key in sorted(all_results.keys()):
            all_results_tmp.extend(all_results[key])
        all_results = all_results_tmp

    file_utils.write_pickle_file(all_results, output_file_path)


def merge(
    dataset_path: str, output_dir_path: str, total: int, index_type: str, prefix: str
) -> None:
    output_file_path = os.path.join(
        output_dir_path, get_file_name(prefix=prefix, index_type=index_type)
    )
    # List all the files in the directory
    file_names = os.listdir(output_dir_path)
    file_names = fileter_file_names(file_names, total)
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
    logger.info(f"Loding the dataset from {dataset_path}")
    dataset: List = file_utils.read_csv_file(
        dataset_path, delimiter="\t", first_row_as_header=True
    )

    # Flatten list
    all_values: List[List[Tuple[int, int]]] = []
    for key, values in all_dict.items():
        all_values.extend(values)

    # Perform further processing for empty values
    assert len(all_values) == len(
        dataset
    ), f"Length of all_values: {len(all_values)} != Length of dataset: {len(dataset)}"
    empty_indices = [i for i, values in enumerate(all_values) if len(values) == 0]

    if empty_indices:
        raise ValueError(
            f"Found {len(empty_indices)} empty indices. Please run extract operation first."
        )

    # Save the dictionary
    logger.info(
        f"Saving the dictionary of length {len(all_values)} to {output_file_path}"
    )
    file_utils.write_pickle_file(all_values, output_file_path)
    return None


def filter(output_dir_path: str, index_type: str, prefix: str) -> None:
    output_file_path = os.path.join(
        output_dir_path, get_file_name(prefix=prefix, index_type=index_type)
    )
    logger.info(f"Loading the phrase indices from {output_file_path}")
    data = file_utils.read_pickle_file(output_file_path)
    for i in tqdm.tqdm(range(len(data))):
        datum = data[i]
        data[i] = [value for value in datum if value[1] - value[0] > 1]
    logger.info(f"Writing the filtered phrase indices to {output_file_path}")
    file_name = get_file_name(prefix=prefix, index_type=index_type).replace(
        "indices", "filtered_indices"
    )
    filtered_output_file_path = os.path.join(output_dir_path, file_name)
    file_utils.write_pickle_file(data, filtered_output_file_path)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg) -> None:
    """
    Args:
        - op: merge, filter, extract
        - target_data: query, document
        - index_type: word, phrase
        - i (optional): split index
        - total: total number of splits
    """
    if cfg.target_data == "query":
        prefix = "query"
        dataset_path = cfg.dataset.query_file
        tokenizer_cfg = cfg.q_tokenizer
    elif cfg.target_data == "document":
        prefix = "doc"
        dataset_path = cfg.dataset.corpus_file
        tokenizer_cfg = cfg.d_tokenizer
    else:
        raise ValueError(f"Invalid type: {cfg.type}")

    if cfg.op == "merge":
        merge(
            dataset_path=dataset_path,
            output_dir_path=cfg.dir_path,
            total=cfg.total,
            index_type=cfg.index_type,
            prefix=prefix,
        )
    elif cfg.op == "filter":
        filter(output_dir_path=cfg.dir_path, index_type=cfg.index_type, prefix=prefix)
    elif cfg.op == "extract":
        extract(
            dir_path=os.path.join(cfg.dataset.dir_path, cfg.dataset.name),
            dataset_path=os.path.join(
                cfg.dataset.dir_path, cfg.dataset.name, dataset_path
            ),
            tokenizer_cfg=tokenizer_cfg,
            split_i=cfg.i,
            total=cfg.total,
            index_type=cfg.index_type,
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
