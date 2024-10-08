import logging
import os

import hkkang_utils.file as file_utils
import hydra
from omegaconf import DictConfig

from eagle.phrase.utils import get_output_file_name
from eagle.tokenization import Tokenizer, Tokenizers

logger = logging.getLogger("AnalysisExtractedPhrase")


def examine_extracted_phrase_ranges(
    cfg: DictConfig, tokenizer: Tokenizer, prefix: str, dataset_path: str
) -> None:
    # Load data
    logger.info(f"Loading dataset from {dataset_path} ...")
    dataset = file_utils.read_json_file(dataset_path, auto_detect_extension=True)

    # Load extracted phrases
    phrase_file_path = os.path.join(
        cfg.dataset.dir_path,
        cfg.dataset.name,
        get_output_file_name(prefix=prefix, total_process_num=0, process_idx=0),
    )
    logger.info(f"Loading extracted phrases from {phrase_file_path} ...")
    extracted_phrases = file_utils.read_pickle_file(phrase_file_path)

    # Examine the extracted phrases
    for doc_idx, (data, phrase_ranges) in enumerate(zip(dataset, extracted_phrases)):
        sentences = data["text"]
        assert len(sentences) == len(
            phrase_ranges
        ), f"{len(sentences)} vs {len(phrase_ranges)}"
        # Tokenize the sentences
        tokenized_ids_in_sentences = [
            tokenizer.tokenize(sent).ids for sent in sentences
        ]
        # Convert tokenized id to token text
        tokenized_toks_in_sentences = [
            tokenizer.tokenizer.convert_ids_to_tokens(sent, skip_special_tokens=False)
            for sent in tokenized_ids_in_sentences
        ]
        # Check the extracted phrases
        for sent_idx, (tokenized_toks, phrase_range) in enumerate(
            zip(tokenized_toks_in_sentences, phrase_ranges)
        ):
            logger.info(f"Data idx: {doc_idx} Sentence idx: {sent_idx}")
            logger.info(f"Text: {sentences[sent_idx]}")
            # Print the extracted phrases
            for phraes_idx, (phrase_start, phrase_end) in enumerate(phrase_range):
                logger.info(
                    f"Phrase: {tokenized_toks[phrase_start:phrase_end]} (idx: {phraes_idx})"
                )
            logger.info("=" * 100)
            # Receive the user input
            stop = input("Press Enter to continue, or type 'stop' to stop: ")
            if "stop" in stop:
                logger.info("Stop the process")
                return None
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)

    # Prepare tokenizers
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )

    # Examine the extracted phrases for query
    logger.info(f"Examine the extracted phrases for query")
    dataset_path = os.path.join(dir_path, cfg.dataset.query_file)
    examine_extracted_phrase_ranges(
        cfg, tokenizer=tokenizers.q_tokenizer, prefix="query", dataset_path=dataset_path
    )

    # Examine the extracted phrases for document
    logger.info(f"Examine the extracted phrases for document")
    dataset_path = os.path.join(dir_path, cfg.dataset.corpus_file)
    examine_extracted_phrase_ranges(
        cfg,
        tokenizer=tokenizers.d_tokenizer,
        prefix="document",
        dataset_path=dataset_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logger.info("Done!")
