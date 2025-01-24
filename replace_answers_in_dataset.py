import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from scripts.analysis.utils import DATASET_NAMES

logger = logging.getLogger("ReplaceAnswers")


def change_answers_format_in_dataset(dataset_path: str) -> None:
    # Read
    logger.info(f"Reading from {dataset_path}")
    dataset: List[Dict] = file_utils.read_jsonl_file(dataset_path)
    for item in tqdm.tqdm(dataset):
        item["answers"] = {"1": item["answers"]}
    logger.info(f"Writing to {dataset_path}")
    file_utils.write_jsonl_file(dataset, dataset_path)
    return dataset


def fix_answers_format_in_dataset(dataset_path: str) -> None:
    # Read
    logger.info(f"Reading from {dataset_path}")
    dataset: List[Dict] = file_utils.read_jsonl_file(dataset_path)

    for item in tqdm.tqdm(dataset):
        assert (
            len(item["answers"]) == 1
        ), f"Expected 1 answer, got {len(item['answers'])}"
        item["answers"] = item["answers"][0]["1"][0]
    logger.info(f"Writing to {dataset_path}")
    file_utils.write_jsonl_file(dataset, dataset_path)
    return dataset


def replace_answers_in_dataset(src_data_path: str, dst_data_path: str) -> None:
    # Read in src data
    logger.info(f"Reading from {SRC_DATA_PATH}")
    src_data: List[Dict] = file_utils.read_jsonl_file(SRC_DATA_PATH)

    # Read in dst data
    logger.info(f"Reading from {DST_DATA_PATH}")
    dst_data: List[Dict] = file_utils.read_jsonl_file(DST_DATA_PATH)

    # Replace "answers" key of each items in the dst_data list with one in the src_data
    for dst_item, src_item in tqdm.tqdm(
        zip(dst_data, src_data, strict=True), total=len(dst_data)
    ):
        dst_item["answers"] = src_item["answers"]

    # Write to dst data
    logger.info(f"Writing to {DST_DATA_PATH}")
    file_utils.write_jsonl_file(dst_data, DST_DATA_PATH)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # replace_answers_in_dataset(SRC_DATA_PATH, DST_DATA_PATH)
    # fix_answers_format_in_dataset(DST_DATA_PATH)
    for DATASET_NAME in tqdm.tqdm(DATASET_NAMES, desc="Dataset"):
        DST_DATA_PATH = f"/root/EAGLE/data/{DATASET_NAME}/dev.jsonl"
        change_answers_format_in_dataset(DST_DATA_PATH)
    logger.info("Done")
