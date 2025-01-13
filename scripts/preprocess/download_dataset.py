import argparse
import logging
import math
import os
import shutil
import tarfile
from typing import *

import hkkang_utils.file as file_utils
import tqdm
import wget
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader

from eagle.phrase.clean import unidecode_text
from scripts.utils import BEIR_DATASET_NAMES

logger = logging.getLogger("DownloadDataset")

LOTTE_SUB_DATASET_NAMES = [
    "lifestyle",
    "pooled",
    "recreation",
    "science",
    "technology",
    "writing",
]
BIER_URL_PREFIX = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
LOTTE_URL = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz"

DOWNLOAD_DIR_PATH = "/root/EAGLE/data/tmp/"


def bar_custom(current, total, width=80):
    width = 30
    avail_dots = width - 2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = "[" + "■" * shaded_dots + " " * (avail_dots - shaded_dots) + "]"
    progress = "%d%% %s [%d / %d]" % (
        current / total * 100,
        percent_bar,
        current,
        total,
    )
    return progress


def wget_download(url, out_path="."):
    # Create directory if not exists
    os.makedirs(out_path, exist_ok=True)
    wget.download(url, out=out_path, bar=bar_custom)


def move_files_to_parent(directory: str) -> None:
    parent_directory = os.path.dirname(directory)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        shutil.move(file_path, parent_directory)

    # Remove the now empty directory
    os.rmdir(directory)


def clean_corpus(corpus: Dict[str, Dict[str, str]], output_path: str) -> None:
    collection = []
    mapping = {}
    for idx, (key, value) in enumerate(
        tqdm.tqdm(corpus.items(), desc="Cleaning corpus"), start=1
    ):
        tmp = {
            "_id": idx,
            "text": unidecode_text(value["text"]),
            "title": unidecode_text(value["title"]),
        }
        mapping[key] = idx
        collection.append(tmp)
    logger.info(f"Saving collection file...")
    file_utils.write_jsonl_file(collection, output_path)
    logger.info(f"Saved {len(collection)} data as {output_path}")
    return mapping


def save_dev_file(
    queries: List, qrels: Dict, mapping: Dict[str, int], output_path: str
) -> None:
    logger.info(f"Saving dev file as {output_path}...")
    dev = []
    for key, query in queries.items():
        answers: Dict[str, int] = qrels[key]

        # Build inverted index
        answer_iv_index: Dict[int, List[int]] = {}
        for str_key, answer_id in answers.items():
            answer_id = int(answer_id)
            # Don't include the answer with score 0
            if answer_id == 0:
                continue
            if answer_id not in answer_iv_index:
                answer_iv_index[answer_id] = []
            if str_key not in mapping:
                raise ValueError(f"Cannot find {str_key} in mapping")
            answer_iv_index[answer_id].append(mapping[str_key])
        dev.append({"id": key, "query": query, "answers": answer_iv_index})
    file_utils.write_jsonl_file(dev, output_path)
    logger.info(f"Saved {len(dev)} data as {output_path}")


def download_dataset(download_path: str, dataset_name: str) -> None:
    assert dataset_name in BEIR_DATASET_NAMES + ["lotte"]
    if dataset_name in BEIR_DATASET_NAMES:
        return download_beir_dataset(download_path, dataset_name)
    else:
        return download_lotte_dataset(download_path)


def download_beir_dataset(download_path: str, dataset_name: str) -> None:
    dataset_name = dataset_name.lower()
    url = os.path.join(BIER_URL_PREFIX, f"{dataset_name}.zip")
    out_dir = os.path.join(download_path, dataset_name)
    logger.info(f"Downloading {dataset_name} dataset from {url} to {out_dir}")
    data_path = util.download_and_unzip(url, out_dir)
    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    logger.info(f"Parsing dataset...")
    mapping = clean_corpus(corpus, output_path=os.path.join(data_path, f"corpus.jsonl"))
    save_dev_file(
        queries, qrels, mapping, output_path=os.path.join(data_path, f"dev.jsonl")
    )
    # Move every files in the directory to the parent directory
    logger.info(f"Clean up the directory {data_path}")
    move_files_to_parent(os.path.join(data_path))
    # Remove the zip file
    os.remove(os.path.join(out_dir, f"{dataset_name}.zip"))
    logger.info(f"Done!")


def download_lotte_dataset(download_path: str) -> None:
    # Download and extract lotte.tar.gz if not exists
    out_dir_path = os.path.join(download_path, "lotte")
    if not os.path.exists(out_dir_path):
        # Download file if not exists
        file_path = os.path.join(download_path, "lotte.tar.gz")
        if not os.path.exists(file_path):
            logger.info(f"Downloading lotte.tar.gz to {download_path}")
            wget_download(LOTTE_URL, out_path=download_path)
            logger.info(f"Downloaded lotte.tar.gz to {download_path}")

        # Extract downloaded file
        logger.info(f"Extracting lotte.tar.gz to {download_path} ...")
        file = tarfile.open(file_path)
        file.extractall(download_path)
        file.close()
        logger.info(f"Extracted lotte.tar.gz to {download_path} !")
        # Remove the tar file
        os.remove(file_path)

    # Parse dataset
    logger.info(f"Parsing LOTTE dataset...")
    for sub_dataset_name in tqdm.tqdm(
        LOTTE_SUB_DATASET_NAMES, desc="Parsing each sub dataset"
    ):
        for split_name in ["dev", "test"]:
            # Create corpus.jsonl
            file_path = os.path.join(
                out_dir_path, sub_dataset_name, split_name, "collection.tsv"
            )
            logger.info(f"Reading {file_path} ...")
            collection: List[Tuple] = file_utils.read_csv_file(
                file_path, delimiter="\t", first_row_as_header=False
            )
            corpus: List[Dict[str, Any]] = []
            for item in collection:
                corpus.append(
                    {
                        "_id": item[0],
                        "text": item[1],
                        "title": "",
                    }
                )
            # Write corpus.jsonl
            corpus_path = os.path.join(
                out_dir_path, sub_dataset_name, split_name, "corpus.jsonl"
            )
            logger.info(f"Writing {len(corpus)} data to {corpus_path} ...")
            file_utils.write_jsonl_file(corpus, corpus_path)
            logger.info(f"Removing {file_path} ...")
            os.remove(file_path)

            for data_type in ["forum", "search"]:
                # Create dev.json
                file_path = os.path.join(
                    out_dir_path, sub_dataset_name, split_name, f"qas.{data_type}.jsonl"
                )
                logger.info(f"Reading {file_path} ...")
                qas = file_utils.read_json_file(file_path, auto_detect_extension=True)
                dev_data: List[Dict[str, Any]] = []
                for item in qas:
                    dev_data.append(
                        {
                            "id": item["qid"],
                            "query": item["query"],
                            "answers": item["answer_pids"],
                        }
                    )
                # Write dev.json
                dev_data_path = os.path.join(
                    out_dir_path, sub_dataset_name, split_name, f"{data_type}_dev.jsonl"
                )
                logger.info(f"Writing {len(dev_data)} data to {dev_data_path} ...")
                file_utils.write_jsonl_file(dev_data, dev_data_path)
                logger.info(f"Removing {file_path} ...")
                os.remove(file_path)

                # Create queries.jsonl
                file_path = os.path.join(
                    out_dir_path,
                    sub_dataset_name,
                    split_name,
                    f"questions.{data_type}.tsv",
                )
                logger.info(f"Reading {file_path} ...")
                questions_data = file_utils.read_csv_file(
                    file_path, delimiter="\t", first_row_as_header=False
                )
                queries: List[Dict[str, Any]] = []
                for item in questions_data:
                    queries.append(
                        {
                            "_id": item[0],
                            "text": item[1],
                        }
                    )

                # Write queries.jsonl
                queries_path = os.path.join(
                    out_dir_path,
                    sub_dataset_name,
                    split_name,
                    f"{data_type}_queries.jsonl",
                )
                logger.info(f"Writing {len(queries)} data to {queries_path} ...")
                file_utils.write_jsonl_file(queries, queries_path)
                logger.info(f"Removing {file_path} ...")
                os.remove(file_path)

            # Clean up the directory
            # Remove metadata.json
            metadata_path = os.path.join(
                out_dir_path, sub_dataset_name, split_name, "metadata.jsonl"
            )
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

    return None


def main(dataset_names: List[str]):
    #### Download scifact.zip dataset and unzip the dataset
    failed_list = []
    for dataset_name in tqdm.tqdm(dataset_names):
        if os.path.exists(os.path.join(DOWNLOAD_DIR_PATH, dataset_name)):
            logger.info(f"{dataset_name} already exists in {DOWNLOAD_DIR_PATH}")
            continue
        try:
            download_dataset(DOWNLOAD_DIR_PATH, dataset_name)
        except:
            logger.info(f"Failed to download {dataset_name}")
            failed_list.append(dataset_name)

    if failed_list:
        logger.info(f"Failed to download {failed_list}")

    logger.info(f"Done!")


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="Whether to download all datasets"
    )
    parser.add_argument(
        "--dataset", type=str, help="The name of the dataset to download"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    args = arg_parse()

    if args.all:
        main(dataset_names=BEIR_DATASET_NAMES + LOTTE_SUB_DATASET_NAMES)
    elif args.dataset:
        main(dataset_names=[args.dataset])
    else:
        raise ValueError("Either --all or --dataset should be provided")
    logger.info("Done!")
