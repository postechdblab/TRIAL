import functools
import logging
import os
from typing import *

import hkkang_utils.concurrent as concurrent_utils
import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hkkang_utils.slack as slack_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.dataset.utils import read_compressed, save_compressed
from eagle.tokenizer import Tokenizer, Tokenizers

logger = logging.getLogger("TokenizeData")

CHUNK_SIZE = 1000


def merge_wrapper(cfg: DictConfig, total_proc_num: int) -> None:
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    # Merge tokenized query
    logger.info(f"Merging tokenized query...")
    merge(
        cfg=cfg,
        file_name=cfg.dataset.query_file,
        tokenizer_name=tokenizers.q_tokenizer.name,
        total_proc_num=total_proc_num,
    )
    logger.info(f"Merging tokenized document...")
    merge(
        cfg=cfg,
        file_name=cfg.dataset.corpus_file,
        tokenizer_name=tokenizers.d_tokenizer.name,
        total_proc_num=total_proc_num,
    )
    return None


def merge(
    cfg: DictConfig, file_name: str, tokenizer_name: str, total_proc_num: int
) -> None:
    assert (
        total_proc_num > 1
    ), f"total_proc_num must be greater than 1, but got {total_proc_num}"

    # Get path
    all_tokenized_data = []
    for i in range(total_proc_num):
        tokenized_dataset_path = os.path.join(
            cfg.dataset.dir_path,
            cfg.dataset.name,
            f"{file_name}.{cfg.model.backbone_name}-tok.{i}.cache",
        )
        if not os.path.exists(tokenized_dataset_path):
            raise FileNotFoundError(f"{tokenized_dataset_path} does not exist!")
        # Read the tokenized dataset
        logger.info(f"Reading tokenized dataset from {tokenized_dataset_path}")
        tokenized_dataset = read_compressed(tokenized_dataset_path)
        all_tokenized_data.append(tokenized_dataset)
    partial_processor = concurrent_utils.PartialProcessor(total_proc_n=total_proc_num)

    # Merge the tokenized dataset
    logger.info(f"Merging {len(all_tokenized_data)} tokenized chunks...")
    all_tokenized_data = partial_processor.merge(all_tokenized_data)

    # Save the merged tokenized dataset
    merged_tokenized_dataset_path = os.path.join(
        cfg.dataset.dir_path,
        cfg.dataset.name,
        f"{file_name}.{tokenizer_name}-tok.cache",
    )
    logger.info(f"Saving merged tokenized dataset to {merged_tokenized_dataset_path}")
    save_compressed(merged_tokenized_dataset_path, all_tokenized_data)

    # Remove the splitted tokenized dataset
    logger.info(f"Removing {total_proc_num} splitted tokenized chunks...")
    for i in range(total_proc_num):
        tokenized_dataset_path = os.path.join(
            cfg.dataset.dir_path,
            cfg.dataset.name,
            f"{file_name}.{tokenizer_name}-tok.{i}.cache",
        )
        os.remove(tokenized_dataset_path)

    return all_tokenized_data


def tokenize_wrapper(
    cfg: DictConfig, total_proc_num: int, current_proc_idx: int
) -> None:
    # Get tokenizers
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    # Tokenize query
    logger.info(f"Tokenizing query...")
    tokenize_and_save(
        cfg=cfg,
        total_proc_num=total_proc_num,
        current_proc_idx=current_proc_idx,
        file_name=cfg.dataset.query_file,
        tokenizer=tokenizers.q_tokenizer,
    )

    # Tokenize document
    logger.info(f"Tokenizing document...")
    tokenize_and_save(
        cfg=cfg,
        total_proc_num=total_proc_num,
        current_proc_idx=current_proc_idx,
        file_name=cfg.dataset.corpus_file,
        tokenizer=tokenizers.d_tokenizer,
    )
    return None


def tokenize_and_save(
    cfg: DictConfig,
    total_proc_num: int,
    current_proc_idx: int,
    file_name: str,
    tokenizer: Tokenizer,
) -> None:
    # Get partial processor
    partial_processor = concurrent_utils.PartialProcessor(
        total_proc_n=total_proc_num,
        current_proc_n=current_proc_idx,
    )

    # Get dataset path
    dataset_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, file_name)
    tokenized_dataset_path = os.path.join(
        cfg.dataset.dir_path,
        cfg.dataset.name,
        (
            f"{file_name}.{tokenizer.name}-tok.cache"
            if total_proc_num == 1
            else f"{file_name}.{tokenizer.name}-tok.{current_proc_idx}.cache"
        ),
    )

    logger.info(f"Reading dataset from {dataset_path}")
    dataset = file_utils.read_json_file(dataset_path, auto_detect_extension=True)

    # Get the target chunk
    texts = list_utils.do_flatten_list([t["text"] for t in dataset])
    text_lens = [len(t["text"]) for t in dataset]
    logger.info("Tokenizing texts...")
    tokenized_texts = partial_processor(
        data=texts, func=functools.partial(tokenizer.tokenize_batch, truncation=False)
    )
    logger.info(f"Tokenized {len(texts)} texts")
    tok_ids = tokenized_texts["input_ids"]
    text_lens_chunk = partial_processor.get_partial_data(data=text_lens)

    # Convert back to the original format
    start_idx = 0
    all_tok_ids = []
    for text_len in tqdm.tqdm(text_lens_chunk):
        all_tok_ids.append(tok_ids[start_idx : start_idx + text_len])
        start_idx += text_len

    # Save the tokenized query dataset
    logger.info(f"Saving {len(all_tok_ids)} tokenized data to {tokenized_dataset_path}")
    save_compressed(tokenized_dataset_path, all_tok_ids)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Args:
        - op: merge, tokenize
        - i (optional): split index
        - total: total number of splits
    """
    # Get output directory
    if cfg.op == "merge":
        merge_wrapper(
            cfg=cfg,
            total_proc_num=cfg.total,
        )
    elif cfg.op == "tokenize":
        tokenize_wrapper(
            cfg=cfg,
            total_proc_num=cfg.total,
            current_proc_idx=cfg.i,
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
