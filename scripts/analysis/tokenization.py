import argparse
import logging

from scripts.utils import load_tokenizer

logger = logging.getLogger("Tokenization")


def main(sentence: str, is_for_query: bool = False) -> None:
    # Create tokenizer
    tokenizer = load_tokenizer(
        is_for_query=is_for_query, model_name="bert-base-uncased"
    )

    # Tensorize
    ids, mask = tokenizer.tensorize([sentence])

    # Convert ids to tokens
    tokens = tokenizer.tok.convert_ids_to_tokens(ids[0])

    logger.info(f"Input: {sentence}")
    logger.info(f"Tokens: {tokens}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check tokenization of a sentence.")
    parser.add_argument("--text", type=str, help="Input text to tokenize.")
    parser.add_argument(
        "--query", action="store_true", help="Whether the input is a query."
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()

    main(args.text, is_for_query=bool(args.query))

    logger.info("Done!")
