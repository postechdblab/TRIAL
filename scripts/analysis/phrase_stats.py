import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import concurrent.futures
import json
import logging
import multiprocessing
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.phrase.extraction import PhraseExtractor
from eagle.tokenization.tokenizers import Tokenizers
from scripts.analysis.utils import DATASET_NAMES, avg

multiprocessing.set_start_method("spawn", force=True)  # Set 'spawn' start method
logger = logging.getLogger("PhraseStats")


def get_stats_for_dataset(
    phrase_extractor: PhraseExtractor,
    dataset: List[Dict[str, Any]],
    dataset_name: str = "",
    sample_num: int = 10000,
) -> Dict[str, int]:

    num_phrases_list: List[int] = []
    num_token_per_phrases_list: List[int] = []
    with tqdm.tqdm(
        total=min(sample_num, len(dataset)), desc=f"{dataset_name} Data"
    ) as pbar:
        cnt = 0
        for datum in dataset:
            if cnt >= sample_num:
                break
            text = datum["text"]
            if isinstance(text, list):
                full_paragraph_text = " ".join(text)
            else:
                full_paragraph_text = text
            if len(full_paragraph_text) == 0:
                continue
            try:
                phrases = phrase_extractor(full_paragraph_text, to_token_indices=True)
            except:
                raise ValueError(
                    f"Error with dataset {dataset_name}. While extracting phrases from {full_paragraph_text}"
                )
            num_phrases_list.append(len(phrases))
            num_token_per_phrases_list.extend(
                [ranges[1] - ranges[0] for ranges in phrases]
            )
            pbar.update(1)
            cnt += 1

    return {
        "avg_num_phrases_per_doc": avg(num_phrases_list),
        "max_num_phrases_per_doc": max(num_phrases_list),
        "min_num_phrases_per_doc": min(num_phrases_list),
        "avg_tok_per_phrases": avg(num_token_per_phrases_list),
        "max_tok_per_phrases": max(num_token_per_phrases_list),
        "min_tok_per_phrases": min(num_token_per_phrases_list),
        "total_samples": len(num_phrases_list),
    }


def process_dataset(
    dataset_name: str, query_path: str, corpus_path: str, cfg: DictConfig
) -> Dict[str, Dict[str, int]]:

    # Initialize tokenizers and extractors in each process
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    q_extractor = PhraseExtractor(tokenizer=tokenizers.q_tokenizer)
    d_extractor = PhraseExtractor(tokenizer=tokenizers.d_tokenizer)

    logger.info(f"Loading the {dataset_name} query dataset from {query_path} ...")
    query_dataset: List = file_utils.read_json_file(
        query_path, auto_detect_extension=True
    )
    logger.info(f"Loading the {dataset_name} document dataset from {corpus_path} ...")
    corpus_dataset: List = file_utils.read_json_file(
        corpus_path, auto_detect_extension=True
    )

    logger.info(f"Getting the phrase statistics for {dataset_name} query dataset")
    query_stats = get_stats_for_dataset(q_extractor, query_dataset, dataset_name)

    logger.info(f"Getting the phrase statistics for {dataset_name} document dataset")
    doc_stats = get_stats_for_dataset(d_extractor, corpus_dataset, dataset_name)

    return {f"{dataset_name}_query": query_stats, f"{dataset_name}_doc": doc_stats}


def get_stats_for_all_dataset(
    cfg: DictConfig, dataset_names: List[str]
) -> Dict[str, Dict[str, int]]:

    dataset_query_paths = []
    dataset_corpus_paths = []
    for dataset_name in dataset_names:
        dir_path = os.path.join(cfg.dataset.dir_path, dataset_name)
        dataset_query_paths.append(os.path.join(dir_path, cfg.dataset.query_file))
        dataset_corpus_paths.append(os.path.join(dir_path, cfg.dataset.corpus_file))

    stats: Dict[str, Dict[str, int]] = {}

    if "multiprocessing" in cfg and cfg.multiprocessing:
        # Parallel processing using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_dataset, dataset_name, query_path, corpus_path, cfg
                )
                for dataset_name, query_path, corpus_path in zip(
                    dataset_names, dataset_query_paths, dataset_corpus_paths
                )
            ]

            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Dataset",
            ):
                result = future.result()
                stats.update(result)
    else:
        # Sequential processing
        for dataset_name, query_path, corpus_path in zip(
            dataset_names, dataset_query_paths, dataset_corpus_paths
        ):
            result = process_dataset(dataset_name, query_path, corpus_path, cfg)
            stats.update(result)

    return stats


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    all_stats = get_stats_for_all_dataset(cfg=cfg, dataset_names=DATASET_NAMES)
    logger.info(json.dumps(all_stats, indent=4))
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
