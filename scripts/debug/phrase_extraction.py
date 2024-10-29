import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.dataset.utils import read_compressed
from eagle.phrase.constituency import ConstituencyParser
from eagle.phrase.extraction import PhraseExtractor
from eagle.phrase.utils import get_tokenized_path
from eagle.tokenization.tokenizers import Tokenizers

logger = logging.getLogger("DebugPhraseExtraction")


def examine_phrase_extraction_from_dataset(
    cfg: DictConfig,
) -> None:
    # Load data
    data_path = "/root/EAGLE/data/beir-msmarco/corpus.jsonl"
    logger.info(f"Loading dataset from {data_path} ...")
    dataset: List[Dict] = file_utils.read_json_file(
        data_path, auto_detect_extension=True
    )
    logger.info(f"Dataset size: {len(dataset)}")

    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    extractor = PhraseExtractor(tokenizer=tokenizers.d_tokenizer)

    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)
    tokenized_path = get_tokenized_path(
        tokenizer=tokenizers.d_tokenizer,
        dir_path=dir_path,
        filename=cfg.dataset.corpus_file,
    )

    tokenized_data: Dict = read_compressed(tokenized_path)

    for datum in dataset:
        texts = datum["text"]
        tokenized_text = tokenized_data[datum["_id"]]
        result: List[List[Tuple[int, int]]] = extractor(
            texts=texts,
            tok_ids_list=tokenized_text,
            to_token_indices=True,
        )
        # Convert token ids to tokens
        sents_in_tokens: List[List[str]] = [
            tokenizers.d_tokenizer.tokenizer.convert_ids_to_tokens(item)
            for item in tokenized_text
        ]
        for idx in range(len(texts)):
            tokens = sents_in_tokens[idx]
            spans = result[idx]
            phrases = []
            for start, end in spans:
                phrase = " ".join(tokens[start:end])
                phrases.append(phrase)
            # Debug
            logger.info(f"Text: {texts[idx]}")
            logger.info(f"Phrases: {phrases}\n")
        print("\n")
        tmp = input("Continue? [y/n]")
        if tmp.lower() == "n":
            break

    return None


def examine_phrase_extraction_from_live_input(
    cfg: DictConfig,
) -> None:
    # Load data
    data_path = "/root/EAGLE/data/beir-msmarco/corpus.jsonl"
    logger.info(f"Loading dataset from {data_path} ...")
    dataset: List[Dict] = file_utils.read_json_file(
        data_path, auto_detect_extension=True
    )
    logger.info(f"Dataset size: {len(dataset)}")

    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    extractor = PhraseExtractor(tokenizer=tokenizers.d_tokenizer)

    while True:
        # Read input
        text = input("Enter text: ")
        if text == "exit":
            logger.info("Exiting ...")
            break
        tokenized_text = tokenizers.d_tokenizer(text).ids
        result: List[Tuple[int, int]] = extractor(
            texts=[text],
            tok_ids_list=[tokenized_text],
            to_token_indices=True,
        )[0]
        # Convert token ids to tokens
        tokens = tokenizers.d_tokenizer.tokenizer.convert_ids_to_tokens(tokenized_text)
        phrases = []
        for start, end in result:
            phrase = " ".join(tokens[start:end])
            phrases.append(phrase)
        # Debug
        logger.info(f"Text: {text}")
        logger.info(f"Phrases: {phrases}\n")
        print("\n")

    return None


def examine_constituency_parser() -> None:
    parser = ConstituencyParser()
    # Load data
    data_path = "/root/EAGLE/data/beir-msmarco/corpus.jsonl"
    logger.info(f"Loading dataset from {data_path} ...")
    dataset = file_utils.read_json_file(data_path, auto_detect_extension=True)
    logger.info(f"Dataset size: {len(dataset)}")
    # Parse
    results = []
    for datum in tqdm.tqdm(dataset[:100]):
        texts = datum["text"]
        parsed_sentences = parser(texts=texts, show_progress=True)
        results.append(parsed_sentences)
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    examine_phrase_extraction_from_live_input(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
