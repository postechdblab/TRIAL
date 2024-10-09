import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

logger = logging.getLogger("DebugTokenization")


def main() -> None:
    # Define the phrase list
    phrase_list = ["play an active role", "participate actively", "active lifestyle"]

    # Load the model and tokenizer
    model = SentenceTransformer("whaleloops/phrase-bert")
    phrase_bert_tokenizer = AutoTokenizer.from_pretrained("whaleloops/phrase-bert")
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Encode the phrases into embeddings
    phrase_embs = model.encode(phrase_list)
    [p1, p2, p3] = phrase_embs

    # Function to tokenize user input
    def phrase_bert_tokenize_input(user_input):
        tokens = phrase_bert_tokenizer.tokenize(user_input)
        token_ids = phrase_bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def bert_tokenize_input(user_input):
        tokens = bert_tokenizer.tokenize(user_input)
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    # Example of user input
    # user_input = "I want to participate actively in an active lifestyle."
    while True:
        user_input = input("Enter text: ")
        if user_input == "exit":
            logger.info("Exiting ...")
            break

        phrase_bert_tokens, phrase_bert_token_ids = phrase_bert_tokenize_input(
            user_input
        )
        bert_tokens, bert_token_ids = bert_tokenize_input(user_input)
        assert (
            phrase_bert_tokens == bert_tokens
        ), f"\nPhrase BERT: {phrase_bert_tokens}\nBERT:{bert_tokens}"
        assert (
            phrase_bert_token_ids == bert_token_ids
        ), f"\nPhrase BERT: {phrase_bert_token_ids}\nBERT:{bert_token_ids}"
        stop = 1

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
