import argparse
import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from colbert.noun_extraction.utils import unidecode_text
from scripts.utils import load_tokenizer

logger = logging.getLogger("Validation")


def validate_phrase_indices(
    tokenizer,
    text: str,
    phrase_indices: List[Tuple[int, int]],
    add_special_tokens: bool = False,
) -> bool:
    """visualize phrase in string format and check if the phrase indices are correct."""
    # Tokenize
    tokens: List = tokenizer.tokenize([text], add_special_tokens=add_special_tokens)[0]

    logger.info("Text:")
    logger.info(text)
    logger.info("\nPhrases:")
    for idx, (start_idx, end_idx) in enumerate(phrase_indices):
        phrase = " ".join(tokens[start_idx:end_idx])
        logger.info(f"Phrase {idx}: {phrase}\n")

    value = input("Press Enter to continue...").strip()
    if value == "q":
        exit(0)
    return None


def main(
    phrase_indices_path: str, document_path: str = None, query_path: str = None
) -> None:
    # Load document
    if document_path:
        logger.info(f"Loading documents from {document_path}...")
        texts: List[Tuple[str, str]] = file_utils.read_csv_file(
            document_path, delimiter="\t", first_row_as_header=False
        )
    elif query_path:
        logger.info(f"Loading queries from {query_path}...")
        if query_path.endswith(".tsv"):
            texts: List[Tuple[str, str]] = file_utils.read_csv_file(
                query_path, delimiter="\t", first_row_as_header=False
            )
        elif query_path.endswith(".json"):
            texts: List[Tuple[str, str]] = [
                [item["id"], item["question"]]
                for item in file_utils.read_json_file(query_path)
            ]
        else:
            raise ValueError(f"Invalid query_path: {query_path}")
    else:
        raise ValueError("Either document_path or query_path should be provided.")

    # Load phrase indices
    logger.info(f"Loading phrase indices from {phrase_indices_path}...")
    all_phrase_indices: Dict[str, List[List[Tuple[int, int]]]] = (
        file_utils.read_pickle_file(phrase_indices_path)
    )

    # Convert to dict
    logger.info(f"Converting documents to dict...")
    texts: Dict[str, str] = {text_id: text for text_id, text in texts}

    tokenizer = load_tokenizer(is_for_query=bool(query_path))

    # Validate
    for doc_id, phrase_indices in tqdm.tqdm(all_phrase_indices.items()):
        text = texts[doc_id]
        text = unidecode_text(text)
        validate_phrase_indices(tokenizer, text, phrase_indices)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cached phrase indices.")
    parser.add_argument(
        "--phrase", type=str, help="Path for the cached phrase indices."
    )
    parser.add_argument("--document", type=str, help="Path for the document file.")
    parser.add_argument("--query", type=str, help="Path for the query file.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    main(
        phrase_indices_path=args.phrase,
        document_path=args.document,
        query_path=args.query,
    )
    logger.info(f"Done!")
