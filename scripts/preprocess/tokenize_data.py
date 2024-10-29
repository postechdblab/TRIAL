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

from eagle.dataset.utils import save_compressed
from eagle.tokenization import Tokenizer, Tokenizers
from eagle.phrase.utils import (
    SPLIT_DIR_NAME,
    get_partial_data_name,
)

logger = logging.getLogger("TokenizeData")

CHUNK_SIZE = 1000


def split_and_save_wrapper(
    cfg: DictConfig, total_proc_num: int, start_idx: int = 0, end_idx: int = None
) -> None:
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)

    logger.info(f"Split and save files for query")
    split_and_save_file(
        dir_path=dir_path,
        file_name=cfg.dataset.query_file,
        total_proc_num=total_proc_num,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    logger.info(f"Split and save files for document")
    split_and_save_file(
        dir_path=dir_path,
        file_name=cfg.dataset.corpus_file,
        total_proc_num=total_proc_num,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    logger.info("Done!")


def split_and_save_file(
    dir_path: str,
    file_name: str,
    total_proc_num: int,
    start_idx: int = 0,
    end_idx: int = None,
) -> None:
    save_dir = os.path.join(dir_path, SPLIT_DIR_NAME)

    # Split the corpus file into multiple files
    logger.info(f"Splitting the corpus file into {total_proc_num} files...")
    if end_idx == None:
        logger.info(f"End index is not provided. Set it to {total_proc_num}")
        end_idx = total_proc_num

    # Check if there are already splitted files saved in the directory
    # Get those which are not saved yet
    indices_to_save = []
    for i in range(start_idx, end_idx + 1):
        file_path = get_partial_data_name(
            dir_path=save_dir,
            file_name=file_name,
            total_proc_num=total_proc_num,
            i=i,
        )
        if not os.path.exists(file_path):
            indices_to_save.append(i)

    if len(indices_to_save) == 0:
        logger.info(f"All files are already splitted and saved. Skip.")
        return None
    else:
        logger.info(f"Files to save: {indices_to_save}")

    # Load the corpus
    dataset_path = os.path.join(dir_path, file_name)
    dataset: List = file_utils.read_json_file(dataset_path, auto_detect_extension=True)
    logger.info(f"Loaded {len(dataset)} text.")

    # Split the corpus
    partial_processor = concurrent_utils.PartialProcessor(total_proc_n=total_proc_num)
    dataset_chunks: Generator = partial_processor.divide_into_chunks(dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the splitted files
    for i, chunk in enumerate(dataset_chunks):
        # Skip if the file is already saved
        if i not in indices_to_save:
            continue

        # Get the file names
        file_path = get_partial_data_name(
            dir_path=save_dir,
            file_name=file_name,
            total_proc_num=total_proc_num,
            i=i,
        )

        # Check if file already exists and write the data if not
        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists. Skip.")
        else:
            logger.info(f"Saving the {len(chunk)} texts to {file_path}")
            file_utils.write_pickle_file(chunk, file_path)

    return None


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
        tokenizer_name=tokenizers.q_tokenizer.model_name,
        total_proc_num=total_proc_num,
    )
    logger.info(f"Merging tokenized document...")
    merge(
        cfg=cfg,
        file_name=cfg.dataset.corpus_file,
        tokenizer_name=tokenizers.d_tokenizer.model_name,
        total_proc_num=total_proc_num,
    )
    return None


def merge(
    cfg: DictConfig, file_name: str, tokenizer_name: str, total_proc_num: int
) -> None:
    assert (
        total_proc_num > 1
    ), f"total_proc_num must be greater than 1, but got {total_proc_num}"

    # Get directory path
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, SPLIT_DIR_NAME)
    file_name = f"{file_name}.{tokenizer_name}-tok.cache"

    # Get path
    all_tokenized_data = {}
    for i in range(total_proc_num):
        tokenized_dataset_path = get_partial_data_name(
            dir_path, file_name, total_proc_num=total_proc_num, i=i
        )
        if not os.path.exists(tokenized_dataset_path):
            raise FileNotFoundError(f"{tokenized_dataset_path} does not exist!")

        # Read the tokenized dataset
        logger.info(f"Reading tokenized dataset from {tokenized_dataset_path}")
        tokenized_dataset: Dict[str, List[int]] = file_utils.read_pickle_file(
            tokenized_dataset_path
        )

        # Merge the tokenized dataset
        logger.info(f"Merging {len(all_tokenized_data)} tokenized chunks...")
        for key, value in tokenized_dataset.items():
            if key in all_tokenized_data:
                logger.error(
                    f"Repeated key: {key}. \nOld: {all_tokenized_data[key]}\n New: {value}"
                )
                exit(-1)
            all_tokenized_data[key] = value

    # Save the merged tokenized dataset
    merged_tokenized_dataset_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, file_name
    )
    logger.info(f"Saving merged tokenized dataset to {merged_tokenized_dataset_path}")
    save_compressed(merged_tokenized_dataset_path, all_tokenized_data)

    # Remove the splitted tokenized dataset
    # logger.info(f"Removing {total_proc_num} splitted tokenized chunks...")
    # for i in range(total_proc_num):
    #     tokenized_dataset_path = get_partial_data_name(
    #         dir_path, file_name, total_proc_num=total_proc_num, i=i
    #     )
    #     os.remove(tokenized_dataset_path)

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
    is_partial_data = total_proc_num > 1
    # Get partial processor
    partial_processor = concurrent_utils.PartialProcessor(
        total_proc_n=total_proc_num,
        current_proc_n=current_proc_idx,
    )

    # Get dir path and file names
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)
    if is_partial_data:
        dir_path = os.path.join(dir_path, SPLIT_DIR_NAME)
    output_file_name = f"{file_name}.{tokenizer.model_name}-tok.cache"
    # Get input and output dataset paths
    if is_partial_data:
        dataset_path = get_partial_data_name(
            dir_path=dir_path,
            file_name=file_name,
            total_proc_num=total_proc_num,
            i=current_proc_idx,
        )
        tokenized_output_path = get_partial_data_name(
            dir_path=dir_path,
            file_name=output_file_name,
            total_proc_num=total_proc_num,
            i=current_proc_idx,
        )
    else:
        dataset_path = os.path.join(dir_path, file_name)
        tokenized_output_path = os.path.join(dir_path, output_file_name)

    logger.info(f"Reading dataset from {dataset_path}")

    # Load dataset
    if is_partial_data:
        dataset = file_utils.read_pickle_file(dataset_path)
        tok_ids: Dict = flat_and_tokenize(sub_dataset=dataset, tokenizer=tokenizer)
    else:
        dataset = file_utils.read_json_file(dataset_path, auto_detect_extension=True)
        tok_ids: Dict = partial_processor(
            data=dataset,
            func=functools.partial(flat_and_tokenize, tokenizer=tokenizer),
        )

    # Save the tokenized query dataset
    logger.info(f"Saving {len(tok_ids)} tokenized data to {tokenized_output_path}")
    if is_partial_data:
        file_utils.write_pickle_file(tok_ids, tokenized_output_path)
    else:
        save_compressed(tokenized_output_path, tok_ids)

    return None


def flat_and_tokenize(
    sub_dataset: Dict[str, Any], tokenizer: Tokenizer
) -> Dict[str, List[int]]:
    # Get dataset info
    ids = [d["_id"] for d in sub_dataset]
    texts = [d["text"] for d in sub_dataset]

    # Preprocess
    flat_texts = list_utils.do_flatten_list(texts)
    num_sents = [len(text) for text in texts]

    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = tokenizer.tokenize_batch(texts=flat_texts, truncation=False)

    # Back to the original format
    start_idx = 0
    all_tok_ids = []
    for num_sent in num_sents:
        all_tok_ids.append(
            tokenized_dataset["input_ids"][start_idx : start_idx + num_sent]
        )
        start_idx += num_sent
    assert len(all_tok_ids) == len(texts), f"{len(all_tok_ids)} != {len(texts)}"

    # Convert the result into dictionary
    result: Dict[str, List[int]] = {}
    for _id, tok_ids in zip(ids, all_tok_ids, strict=True):
        result[_id] = tok_ids

    return result


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Args:
        - op: merge, tokenize
        - i (optional): split index
        - indices (optional): indices to extract
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
    elif cfg.op == "split_file":
        split_and_save_wrapper(
            cfg,
            total_proc_num=cfg.total,
            start_idx=cfg.indices[0] if "indices" in cfg and cfg.indices else 0,
            end_idx=cfg.indices[1] if "indices" in cfg and cfg.indices else None,
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
