import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm
from nltk.corpus import stopwords
from spacy.lang.en import stop_words

from colbert.noun_extraction.identify_noun import SpacyModel, Text, Token
from colbert.noun_extraction.utils import unidecode_text
from colbert.utils.utils import stem
from model import RetrievalResult
from scripts.evaluate.utils import load_data

DATASET_DIR = "/root/ColBERT/data"

logger = logging.getLogger("tmp")

nltk_words = stopwords.words("english")
spacy_words = stop_words.STOP_WORDS
all_stop_words = list(set(nltk_words) | spacy_words)


def get_result_score(result) -> float:
    return sum(result.token_scores.max(axis=1))


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
    print("Loading data...")
    # Load result data
    logger.info(f"Reading results cache data")
    result_data: List = file_utils.read_pickle_file(
        "/root/ColBERT/debug/result.nq_baseline_nway32_q4_less_hard_lr2_distill.pkl"
    )
    # Load dev data
    dataset_name = "nq"
    logger.info(f"Loading {dataset_name} data")
    eval_data: List = load_data(dataset_dir=DATASET_DIR, dataset_name=dataset_name)

    # Initialize for counting stats

    cnt_hard_neg: int = 0
    cnt_q_has_all_matched: int = 0
    extracted_token_num: List[int] = []
    q_found_token_num: List[int] = []
    d_found_token_num: List[int] = []

    for i, eval_datum in enumerate(tqdm.tqdm(eval_data)):
        qid, query, gold_pids, gold_p_titles, gold_p_scores = eval_datum
        gold_pids_, pos_results, all_top_results = result_data[qid]
        # Check pos_results
        assert all(
            [str(item.doc.id) in gold_pids for item in pos_results]
        ), f"Positive results mismatch: {pos_results} != {gold_pids}"
        assert set(gold_pids_) == set(
            gold_pids
        ), f"Gold pids mismatch: {gold_pids_} != {gold_pids}"
        pos_results: List[RetrievalResult] = pos_results
        top_results: List[RetrievalResult] = all_top_results
        # top_k_reusults: List[RetrievalResult] = top_results[:10]
        # Get neg_results
        neg_results = [
            item for item in top_results if str(item.doc.id) not in gold_pids
        ]
        min_pos_score = min([get_result_score(item) for item in pos_results])
        hard_neg_results = [
            item for item in neg_results if get_result_score(item) >= min_pos_score
        ]

        gold_docs = [item.doc.text for item in pos_results]
        hard_docs = [item.doc.text for item in hard_neg_results]

        if hard_docs:
            # Check if the gold document contains false negative phrases
            q_has_false_negative, q_found_tokens_list, q_all_tokens = (
                contains_false_negative_phrase_batch(query, gold_docs)
            )
            # Check if any of the less hard documents contain false negative phrases
            d_has_false_negative_list, d_found_tokens_list, d_all_tokens = (
                contains_false_negative_phrase_batch(query, hard_docs)
            )

            # Count stats
            cnt_hard_neg += len(hard_neg_results)
            cnt_q_has_all_matched += int(
                any(item == q_all_tokens for item in q_found_tokens_list)
            )
            extracted_token_num.append(len(q_all_tokens))
            q_found_token_num.append(
                sum(len(item) for item in q_found_tokens_list)
                / len(q_found_tokens_list)
            )
            d_found_token_num.append(
                sum(len(item) for item in d_found_tokens_list)
                / len(d_found_tokens_list)
            )

    # Print stats
    logger.info("Stats:")
    logger.info(f"Number of hard negatives: {cnt_hard_neg}")
    logger.info(f"Number of queries where all matched: {cnt_q_has_all_matched}")
    logger.info(
        f"Average number of extracted tokens: {sum(extracted_token_num) / len(extracted_token_num)}"
    )
    logger.info(
        f"Average number of found tokens in queries: {sum(q_found_token_num) / len(q_found_token_num)}"
    )
    logger.info(
        f"Average number of found tokens in documents: {sum(d_found_token_num) / len(d_found_token_num)}"
    )
    logger.info(f"Done!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # test()
    main()
    logger.info(f"Done!")
