from typing import *

import hkkang_utils.file as file_utils
import tqdm
from nltk.corpus import stopwords
from spacy.lang.en import stop_words

from colbert.noun_extraction.identify_noun import SpacyModel, Text, Token
from colbert.noun_extraction.utils import unidecode_text
from colbert.utils.utils import stem

DATASET_DIR = "/root/ColBERT/data"

nltk_words = stopwords.words("english")
spacy_words = stop_words.STOP_WORDS
all_stop_words = list(set(nltk_words) | spacy_words)


def remove_stopwords(words: List[str]) -> List[str]:
    return [word for word in words if word not in all_stop_words]


def strip_punctuation(s: str) -> str:
    # Remove punctuation at the beginning and end of the string
    while len(s) > 0 and not s[0].isalnum():
        s = s[1:]
    while len(s) > 0 and not s[-1].isalnum():
        s = s[:-1]
    return s


def contains_false_negative_phrase_batch(
    query: str, documents: List[str], version: int = 1
) -> Tuple[Tuple[bool], Tuple[List[str]], List[str]]:
    if version == 1:
        # Extract words from query
        parsed_texts: List[Text] = SpacyModel()([query] + documents)
        q_parsed_text, d_parsed_texts = parsed_texts[0], parsed_texts[1:]
        # Extract informative tokens
        q_informative_tokens: List[Token] = q_parsed_text.informative_tokens
        d_informative_tokens_list: List[List[Token]] = [
            d_parsed_text.informative_tokens for d_parsed_text in d_parsed_texts
        ]
        # Perform stemming
        d_informative_words_list = [
            [stem(token.text) for token in d_informative_tokens]
            for d_informative_tokens in d_informative_tokens_list
        ]

        # Check if any of the query words are not in the document
        found_tokens_list = []
        for d_informative_words in d_informative_words_list:
            not_found_tokens = []
            for q_token in q_informative_tokens:
                q_word = stem(q_token.text)
                if q_word in d_informative_words:
                    not_found_tokens.append(q_token)
            found_tokens_list.append(not_found_tokens)

        # Return True if there are false negatives
        return (
            [len(not_found_tokens) > 0 for not_found_tokens in found_tokens_list],
            found_tokens_list,
            q_informative_tokens,
        )
    else:
        raise NotImplementedError(f"Version {version} not implemented")


def contains_false_negative_phrase(
    query: str, document: str, version: int = 1
) -> Tuple[bool, List[str], List[str]]:
    if version == 1:
        # Extract words from query
        parsed_texts: List[Text] = SpacyModel()([query, document])
        q_parsed_text, d_parsed_text = parsed_texts
        # Extract informative tokens
        q_informative_tokens: List[Token] = q_parsed_text.informative_tokens
        d_informative_tokens: List[Token] = d_parsed_text.informative_tokens
        # Perform stemming
        d_informative_words = [stem(token.text) for token in d_informative_tokens]

        # Check if any of the query words are in the document
        found_tokens = []
        for q_token in q_informative_tokens:
            q_word = stem(q_token.text)
            if q_word in d_informative_words:
                found_tokens.append(q_token)

        # Return True if there are false negatives
        return len(found_tokens) > 0, found_tokens, q_informative_tokens
    else:
        raise NotImplementedError(f"Version {version} not implemented")


def test() -> None:
    query = "what was the immediate impact of the success of the manhattan project?"
    document1 = "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated."
    document2 = "The Manhattan Project was a research and development project that produced the first nuclear weapons during World War II.roves appreciated the early British atomic research and the British scientists' contributions to the Manhattan Project, but stated that the United States would have succeeded without them. He also said that Churchill was the best friend the atomic bomb project had [as] he kept Roosevelt's interest up."

    has_false_negative1, phrases1 = contains_false_negative_phrase(query, document1)
    has_false_negative2, phrases2 = contains_false_negative_phrase(query, document2)

    stop = 1


def main(
    train_file_path: str = "/root/ColBERT/data/msmarco_old/train_data_nhards256.jsonl",
    query_file_path: str = "/root/ColBERT/data/msmarco_old/queries.train.tsv",
    collection_file_path: str = "/root/ColBERT/data/msmarco/collection.tsv",
) -> None:
    # Read in training queries
    print("Loading training data...")
    train_data: List[Dict] = file_utils.read_jsonl_file(train_file_path)

    # queries_file_path = "/root/ColBERT/data/msmarco_old/queries.train.tsv"
    queries: List[Tuple[str, str]] = file_utils.read_csv_file(
        query_file_path, delimiter="\t", first_row_as_header=False
    )
    queries: Dict[str, str] = {qid: query for qid, query in queries}

    # Read in collection
    collection: List[Tuple[str, str, str]] = file_utils.read_csv_file(
        collection_file_path, delimiter="\t", first_row_as_header=False, quotechar=None
    )
    collection: List[str] = [doc_text for doc_id, doc_text, doc_title in collection]

    # For counting query stats
    q_all_found_token_cnt: int = 0
    q_all_extracted_token_cnt: List[int] = []
    q_all_has_false_negative_cnt: int = 0
    # For counting document stats
    d_all_found_token_cnt: List[List[int]] = []
    d_all_extracted_token_cnt: List[int] = []
    d_all_has_false_negative_cnt: List[int] = []
    for i, datum in enumerate(tqdm.tqdm(train_data[:10000])):
        qid = datum[0]
        doc_ids = datum[1:]
        query = queries[str(qid)]
        documents = [collection[doc_id + 1] for doc_id in doc_ids]
        gold_doc = unidecode_text(documents[0])
        start_idx = 50
        end_idx = start_idx + 64
        less_hard_documents = [
            unidecode_text(doc) for doc in documents[start_idx:end_idx]
        ]
        # Check if the gold document contains false negative phrases
        q_has_false_negative, q_found_tokens, q_all_tokens = (
            contains_false_negative_phrase(query, gold_doc)
        )
        # Check if any of the less hard documents contain false negative phrases
        d_has_false_negative_list, d_found_tokens_list, d_all_tokens = (
            contains_false_negative_phrase_batch(query, less_hard_documents)
        )

        # Aggregate for query
        q_all_found_token_cnt += int(len(q_found_tokens) == len(q_all_tokens))
        q_all_extracted_token_cnt.append(len(q_all_tokens))
        q_all_has_false_negative_cnt += int(q_has_false_negative)
        # Aggregate for documents
        d_all_found_token_cnt.append([len(item) for item in d_found_tokens_list])
        d_all_extracted_token_cnt.append(len(d_all_tokens))
        d_all_has_false_negative_cnt.append(sum(d_has_false_negative_list))

    print(f"Query with all phrase found: {q_all_found_token_cnt}")
    # Show average number of important tokens in the document
    avg_d_extracted_tokens = sum(d_all_extracted_token_cnt) / len(
        d_all_extracted_token_cnt
    )
    print(f"Average number of extracted tokens:{avg_d_extracted_tokens}")
    avg_d_found_tokens_cnt = [sum(item) / len(item) for item in d_all_found_token_cnt]
    avg_avg_d_found_tokens_cnt = sum(avg_d_found_tokens_cnt) / len(
        avg_d_found_tokens_cnt
    )
    print(f"Aveage found token count: {avg_avg_d_found_tokens_cnt}")

    print(
        f"Document false negative rate: {sum(d_all_has_false_negative_cnt)/len(train_data)}"
    )

    print("Done!")


if __name__ == "__main__":
    # test()
    main()
