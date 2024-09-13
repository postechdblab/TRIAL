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
from eagle.tokenizer import Tokenizer, Tokenizers

logger = logging.getLogger("PhraseExtraction")

CHUNK_SIZE = 1000

SPLIT_DIR_NAME = "splitted"


def remove_file_name_from_path(path: str) -> str:
    return os.path.join("/", *[item for item in path.split("/")[:-1] if item])


def get_partial_data_name(
    dir_path: str, file_name: str, total_proc_num: int, i: int
) -> str:
    return os.path.join(dir_path, f"{file_name}.{i}_{total_proc_num}")


def get_output_file_name(
    prefix: str, total_process_num: int, process_idx: int = None
) -> str:
    return (
        f"phrase_indices.{prefix}.pkl.{process_idx}_{total_process_num}"
        if total_process_num > 1
        else f"phrase_indices.{prefix}.pkl"
    )


def get_tokenized_path(tokenizer: Tokenizer, dir_path: str, filename: str) -> str:
    return os.path.join(dir_path, f"{filename}.{tokenizer.model_name}-tok.cache")


def split_and_save_file(
    cfg: DictConfig, total_proc_num: int, start_idx: int = 0, end_idx: int = None
) -> None:
    # Split the corpus file into multiple files
    logger.info(f"Splitting the corpus file into {total_proc_num} files...")

    if end_idx == None:
        logger.info(f"End index is not provided. Set it to {total_proc_num}")
        end_idx = total_proc_num

    # Check if there are already splitted files saved in the directory
    # Get those which are not saved yet
    indices_to_save = []
    for i in range(start_idx, end_idx):
        file_path = get_partial_data_name(
            dir_path=os.path.join(
                cfg.dataset.dir_path, cfg.dataset.name, SPLIT_DIR_NAME
            ),
            file_name=cfg.dataset.corpus_file,
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

    # Prepare tokenizers
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )

    # Load the corpus
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)
    dataset_path = os.path.join(dir_path, cfg.dataset.corpus_file)
    dataset: List = file_utils.read_json_file(dataset_path, auto_detect_extension=True)
    logger.info(f"Loaded {len(dataset)} text.")

    # Load the tokenized dataset
    tokenized_path = get_tokenized_path(
        tokenizer=tokenizers.d_tokenizer,
        dir_path=dir_path,
        filename=cfg.dataset.corpus_file,
    )
    tokenized_dataset = read_compressed(tokenized_path)

    # Split the corpus
    partial_processor = concurrent_utils.PartialProcessor(total_proc_n=total_proc_num)
    dataset_chunks: Generator = partial_processor.divide_into_chunks(dataset)

    # Split the tokenized dataset
    tokenized_dataset_chunks: Generator = partial_processor.divide_into_chunks(
        tokenized_dataset
    )

    save_dir = os.path.join(dir_path, SPLIT_DIR_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the splitted files
    for i, (chunk, tokenized_chunk) in enumerate(
        zip(dataset_chunks, tokenized_dataset_chunks, strict=True)
    ):
        # Skip if the file is already saved
        if i not in indices_to_save:
            continue

        # Get the file names
        file_path = get_partial_data_name(
            dir_path=save_dir,
            file_name=cfg.dataset.corpus_file,
            total_proc_num=total_proc_num,
            i=i,
        )
        tokenized_file_path = get_partial_data_name(
            dir_path=save_dir,
            file_name=tokenized_path.split("/")[-1],
            total_proc_num=total_proc_num,
            i=i,
        )
        # Check if file already exists and write the data if not
        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists. Skip.")
        else:
            logger.info(f"Saving the {len(chunk)} texts to {file_path}")
            file_utils.write_pickle_file(chunk, file_path)

        # Check if file already exists and write the tokenized data if not
        if os.path.exists(tokenized_file_path):
            logger.info(f"File {tokenized_file_path} already exists. Skip.")
        else:
            # Write the chunk to the file
            logger.info(
                f"Saving the {len(tokenized_chunk)} tokenized texts to {tokenized_file_path}"
            )
            file_utils.write_pickle_file(tokenized_chunk, tokenized_file_path)

    return None


def extract_wrapper(
    cfg: DictConfig, total_proc_num: int, current_proc_idx: int
) -> None:
    # Extract phrase indices for query
    logger.info("Extracting phrase indices for query...")
    # Prepare tokenizers
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    # Get dataset path
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)
    dataset_path = os.path.join(dir_path, cfg.dataset.query_file)
    tokenized_path = get_tokenized_path(
        tokenizer=tokenizers.q_tokenizer,
        dir_path=dir_path,
        filename=cfg.dataset.query_file,
    )
    extract_phrase_indices(
        cfg=cfg,
        dataset_path=dataset_path,
        tokenized_path=tokenized_path,
        split_i=current_proc_idx,
        total=total_proc_num,
        prefix="query",
    )

    logger.info("Extracting phrase indices for document...")
    dataset_path = os.path.join(dir_path, cfg.dataset.corpus_file)
    tokenized_path = get_tokenized_path(
        tokenizer=tokenizers.d_tokenizer,
        dir_path=os.path.join(dir_path, SPLIT_DIR_NAME),
        filename=cfg.dataset.corpus_file,
    )
    tokenized_file_path = get_partial_data_name(
        dir_path="",
        file_name=tokenized_path,
        total_proc_num=total_proc_num,
        i=current_proc_idx,
    )
    extract_phrase_indices(
        cfg=cfg,
        dataset_path=dataset_path,
        tokenized_path=tokenized_file_path,
        split_i=current_proc_idx,
        total=total_proc_num,
        prefix="doc",
        is_splited=True,
    )

    return None


def extract_phrase_indices(
    cfg: DictConfig,
    dataset_path: str,
    tokenized_path: str,
    split_i: int,
    total: int,
    prefix: str,
    is_splited: bool = False,
) -> None:
    logger.info(f"I: {split_i}, Total: {total}")
    partial_processor = concurrent_utils.PartialProcessor(
        total_proc_n=total, current_proc_n=split_i
    )

    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    tokenizer = tokenizers.q_tokenizer if prefix == "query" else tokenizers.d_tokenizer
    extractor = PhraseExtractor(tokenizer=tokenizer)

    # Load tokenized text
    logger.info(f"Loading the tokenizered data from {tokenized_path}")
    if is_splited:
        tokenized_data_chunk = file_utils.read_pickle_file(tokenized_path)
    else:
        tokenized_data = read_compressed(tokenized_path)
        tokenized_data_chunk: List = partial_processor.get_partial_data(tokenized_data)

        # Free up memory by deleting the loaded data except the chunks
        del tokenized_data

    # Load the corpus
    logger.info(f"Loading the text data from {dataset_path}")
    if is_splited:
        data_file_name = dataset_path.split("/")[-1]
        dataset_chunk_path = get_partial_data_name(
            dir_path=os.path.join(
                remove_file_name_from_path(dataset_path), SPLIT_DIR_NAME
            ),
            file_name=data_file_name,
            total_proc_num=total,
            i=split_i,
        )
        dataset_chunk = file_utils.read_pickle_file(dataset_chunk_path)
    else:
        dataset: List = file_utils.read_json_file(
            dataset_path, auto_detect_extension=True
        )
        logger.info(f"Loaded {len(dataset)} text.")

        # Divide the dataset into chunks and extract phrases
        dataset_chunk: List = partial_processor.get_partial_data(dataset)

        # Free up memory by deleting the loaded data except the chunks
        del dataset

    logger.info(f"Target chunk size: {len(dataset_chunk)}")
    mini_dataset_chunks: Generator = list_utils.chunks(dataset_chunk, CHUNK_SIZE)
    mini_tokenized_data_chunks: List = list_utils.chunks(
        tokenized_data_chunk, CHUNK_SIZE
    )

    logger.info(f"Begin to extract phrases from {len(dataset_chunk)} texts")
    all_results: Dict[int, List[List[Tuple[int]]]] = {}
    all_results = []
    for ci, (d_chunk, t_chunk) in enumerate(
        tqdm.tqdm(
            zip(mini_dataset_chunks, mini_tokenized_data_chunks, strict=True),
            total=math.ceil(len(dataset_chunk) / CHUNK_SIZE),
        )
    ):
        # Get all sentences in the text
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

        # Extract phrases
        results = extractor(
            texts=sents,
            tok_ids=tok_ids_in_sent,
            to_token_indices=True,
        )

        # Back to the original text by combining the sentences
        sent_idx = 0
        for sent_len in sent_lens:
            sent_results = results[sent_idx : sent_idx + sent_len]
            all_results.append(sent_results)
            sent_idx += sent_len

        assert sent_idx == len(
            results
        ), f"sent_idx: {sent_idx} != len(results): {len(results)}"

    # Get the output file path
    output_file_name = get_output_file_name(
        prefix=prefix, total_process_num=total, process_idx=split_i
    )
    output_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, SPLIT_DIR_NAME, output_file_name
    )
    logger.info(f"Saving the {len(all_results)} results to {output_file_path}")
    file_utils.write_pickle_file(all_results, output_file_path)


def merge_wrapper(cfg: DictConfig, total_proc_num: int) -> None:
    # Merge the splitted query data
    logger.info("Merging the splitted query data...")
    merge(cfg=cfg, prefix="query", total_process_num=total_proc_num)

    logger.info("Merging the splitted document data...")
    merge(cfg=cfg, prefix="doc", total_process_num=total_proc_num)

    return None


def merge(cfg: DictConfig, prefix: str, total_process_num: int) -> None:
    # Get all the splitted file paths
    file_names = [
        get_output_file_name(
            prefix=prefix, total_process_num=total_process_num, process_idx=process_idx
        )
        for process_idx in range(total_process_num)
    ]
    # Read in all the splitted data
    all_data = []
    for file_name in file_names:
        file_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, file_name)
        data = file_utils.read_pickle_file(file_path)
        all_data.extend(data)

    # Save the merged data
    output_file_path = os.path.join(
        cfg.dataset.dir_path,
        cfg.dataset.name,
        get_output_file_name(prefix=prefix, total_process_num=0, process_idx=0),
    )
    logger.info(f"Saving the merged data to {output_file_path}")
    file_utils.write_pickle_file(all_data, output_file_path)

    # Clean up the splitted files
    logger.info(f"Removing the {len(file_names)} splitted files...")
    for file_name in file_names:
        file_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name, file_name)
        os.remove(file_path)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Args:
        - op: split_file, extract, merge
        - i (optional): split index
        - indices (optional): indices to extract
        - total: total number of splits
    """
    if cfg.op == "split_file":
        split_and_save_file(
            cfg,
            total_proc_num=cfg.total,
            start_idx=cfg.indices[0] if cfg.indices else 0,
            end_idx=cfg.indices[1] if cfg.indices else None,
        )
    elif cfg.op == "merge":
        merge_wrapper(
            cfg=cfg,
            total_proc_num=cfg.total,
        )
    elif cfg.op == "extract":
        extract_wrapper(
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
