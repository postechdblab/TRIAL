import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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

from eagle.phrase.clean import unidecode_text
from eagle.tokenization.sentencizer import Sentencizer

logger = logging.getLogger("SplitSentences")

CHUNK_SIZE = 1000


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Read in the data
    """
    Args:
        - op: merge, split
        - target_data: query, document
        - i (optional): process index
        - total: total number of splits
    """
    # Parse arguments
    if cfg.target_data == "query":
        prefix = "query"
        dataset_path = cfg.dataset.query_file
    elif cfg.target_data == "document":
        prefix = "doc"
        dataset_path = cfg.dataset.corpus_file
    else:
        raise ValueError(f"Invalid type: {cfg.type}")
    dataset_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, dataset_path)

    # Process based on the given operation
    if cfg.op == "split":
        split_text_to_sentences(
            cfg=cfg,
            dataset_path=dataset_path,
            prefix=prefix,
            total_process_num=cfg.total,
            process_idx=cfg.i,
        )
    elif cfg.op == "merge":
        merge_splitted_data(
            cfg=cfg,
            prefix=prefix,
            total_process_num=cfg.total,
        )
    else:
        raise ValueError(f"Invalid operation: {cfg.op}")
    logger.info("Done!")


def merge_splitted_data(cfg: DictConfig, prefix: str, total_process_num: int) -> None:
    # Get all the splitted file paths
    file_names = [
        get_file_name(total_process_num, process_idx, prefix)
        for process_idx in range(total_process_num)
    ]
    # Read in all the splitted data
    all_data = []
    for file_name in file_names:
        file_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, file_name)
        data = file_utils.read_jsonl_file(file_path)
        all_data.extend(data)

    # Save the merged data
    output_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, f"{prefix}_sentences.jsonl"
    )
    logger.info(f"Saving the merged data to {output_file_path}")
    file_utils.write_jsonl_file(all_data, output_file_path)

    # Clean up the splitted files
    logger.info(f"Removing the {len(file_names)} splitted files...")
    for file_name in file_names:
        file_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, file_name)
        os.remove(file_path)
    return None


def get_file_name(total_process_num: int, process_num: int, prefix: str) -> str:
    return (
        f"{prefix}_sentences.{process_num}.jsonl"
        if total_process_num > 1
        else f"{prefix}_sentences.jsonl"
    )


def split_text_to_sentences(
    cfg: DictConfig,
    dataset_path: str,
    prefix: str,
    total_process_num: int,
    process_idx: int,
) -> None:
    # Load model
    sentencizer = Sentencizer()

    # Read in the data
    logger.info(f"Reading data from {dataset_path}")
    dataset = file_utils.read_json_file(dataset_path, auto_detect_extension=True)
    logger.info(f"Data length: {len(dataset)}")

    # Divide the dataset into chunks and extract phrases
    chunks: List = list_utils.divide_into_chunks(dataset, num_chunks=total_process_num)
    target_chunk = chunks[process_idx]
    logger.info(f"Target chunk size: {len(target_chunk)}")
    num_of_chunks = math.ceil(len(target_chunk) / CHUNK_SIZE)

    # Split the text into sentences
    logger.info(f"Parsing data...")
    for chunk in tqdm.tqdm(
        list_utils.chunks(target_chunk, CHUNK_SIZE), total=num_of_chunks
    ):
        texts: List[str] = [unidecode_text(item["text"]) for item in chunk]
        all_sentences: List[List[str]] = sentencizer(texts)
        for i, sentences in enumerate(all_sentences):
            chunk[i]["text"] = sentences

    # Save the parsed data
    output_file_name = get_file_name(total_process_num, process_idx, prefix)
    output_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, output_file_name
    )
    logger.info(f"Saving the parsed data to {output_file_path}")

    file_utils.write_jsonl_file(target_chunk, output_file_path)

    return None


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
