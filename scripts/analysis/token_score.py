import logging
from typing import *

import torch

from colbert.noun_extraction.utils import unidecode_text
from colbert.utils.utils import isint
from model.late_encoder import ColBERTRetriever, RetrievalResult
from model.utils import Document
from scripts.evaluate.utils import get_recall_rates, load_beir_data
from scripts.utils import read_collection

# # Retriever path
# ROOT = "/root/EAGLE/experiments/"
# EXPERIMENT = "msmarco"
# INDEX = "msmarco.nbits=2"
# QUERY_PATH = "/root/EAGLE/data/msmarco/queries.dev.tsv"
# QRELS_PATH = "/root/EAGLE/data/msmarco/qrels.dev.tsv"
# COLLECTION_PATH = "/root/EAGLE/data/msmarco/collection.tsv"

# Retriever path
ROOT = "/root/EAGLE/experiments/"
DATASET_DIR = "/root/EAGLE/data"
CHECKPOINT_DIR = "/root/EAGLE/checkpoint"
NBITS = 2

logger = logging.getLogger("TokenScore")


def load_collection(dataset_name: str) -> Dict:
    collection_path = f"{DATASET_DIR}/{dataset_name}/collection.tsv"
    return read_collection(collection_path)


@torch.no_grad()
def main(
    model_name: str, dataset_name: str, skip_padding: bool, use_phrase_level: bool
) -> None:
    # Read in collection
    collection: Dict[str, str] = load_collection(dataset_name=dataset_name)
    collection = {int(k): v for k, v in collection.items()}

    # initialize retriever
    index = f"{dataset_name}.{model_name}.nbits={NBITS}"
    experiment = f"{dataset_name}_unidecode"
    retriever = ColBERTRetriever(
        root=ROOT,
        index=index,
        experiment=experiment,
        use_cache=True,
        skip_padding=skip_padding,
        is_use_phrase_level=use_phrase_level,
    )

    # Load dataset
    data = load_beir_data(
        dataset_dir=DATASET_DIR, dataset_name=dataset_name, return_unique=True
    )
    # Data dic
    data_dict = {}
    for qid, query, pids, p_titles, scores in data:
        data_dict[qid] = (query, pids, p_titles, scores)

    # Retrieve top-100 passages for each query
    queries = [item[0] for item in data_dict.items()]
    all_results: List[List[RetrievalResult]] = retriever.retrieve_batch(
        queries, return_scores=True
    )

    # Evaluate the retrieval results
    all_recall = {}
    for idx in range(len(all_results)):
        pred_pids = [result.doc.id for result in all_results[idx]]
        recall = get_recall_rates(ranked_pids=pred_pids, gold_pids=[pids[idx]])
        for key, value in recall.items():
            if key not in all_recall:
                all_recall[key] = []
            all_recall[key].append(value)
    # Average the recall rates
    for key, value in all_recall.items():
        all_recall[key] = sum(value) / len(value)
    print(all_recall)

    # Find the wrong indices
    wrong_indices = []
    for idx in range(len(all_results)):
        pred_pids = [result.doc.id for result in all_results[idx]]
        if pids[idx] not in pred_pids[:50]:
            wrong_indices.append(idx)
    print(wrong_indices)

    idx = 37
    query = queries[idx]
    gold_pid = pids[idx]
    pred_docs = [result.doc for result in all_results[idx]]
    pred_pids = [doc.id for doc in pred_docs]
    gold_found = gold_pid in pred_pids
    print(
        f"Is in the gold?: {gold_found}, at {pred_pids.index(gold_pid) if gold_found else -1}"
    )
    show_passages(query=query, docs=pred_docs, gold_pid=gold_pid, collection=collection)

    idx = 37
    query = queries[idx]
    gold_pid = pids[idx]
    examine(retriever=retriever, query=query, pid=gold_pid, collection=collection)


def get_ranks(scores: List[float]) -> List[int]:
    ranks = []
    for i in range(len(scores)):
        rank = 1
        for j in range(len(scores)):
            if scores[j] > scores[i]:
                rank += 1
        ranks.append(rank)
    return ranks


def examine_token_scores(
    q_toks: List[str], d_toks: List[str], tok_scores: List[List[float]]
) -> None:
    tok_scores = torch.tensor(tok_scores)
    # Find the max score
    max_scores = tok_scores.max(dim=1)[0]
    # Show max scores
    for q_i in range(len(q_toks)):
        print(f"Query token {q_i}: {q_toks[q_i]} (max score: {max_scores[q_i]})")
    print("\n")
    for q_i in range(len(q_toks)):
        print(f"Query token {q_i}: {q_toks[q_i]} (max score: {max_scores[q_i]})")
        # Rank the document tokens by score
        ranks = get_ranks(tok_scores[q_i])
        for d_i in range(len(d_toks)):
            print(
                f"\tDocument token {d_i}: {d_toks[d_i]} (score: {tok_scores[q_i][d_i]}, rank: {ranks[d_i]})"
            )
        print("\n")
    return None


# Show gold and non-gold passages
def show_passages(
    query: str, docs: List[Document], gold_pid: int, collection: Dict
) -> None:
    print("Query: ", query)
    print(f"Gold document: (PID: {gold_pid})")
    print(f"{unidecode_text(collection[gold_pid])}\n")
    # Figure out if the gold passage is in the top-100
    gold_idx = -1
    for i, doc in enumerate(docs):
        if doc.id == gold_pid:
            gold_idx = i
            break
    gold_in_top_100 = True if gold_idx != -1 else False
    # Print document texts
    print(f"Gold passage ranked: {gold_idx}")
    for i in range(max(10, gold_in_top_100)):
        doc = docs[i]
        print(f"Document {i+1}: {doc.title} (PID: {doc.id})")
        print(f"{doc.text}\n")


def examine(retriever, query: str, pid: int, collection: Dict) -> None:
    # Tokenizer for query texts
    q_tokenizer = retriever.searcher.checkpoint.query_tokenizer
    # Tokenizer for documents
    d_tokenizer = retriever.searcher.checkpoint.doc_tokenizer

    doc = unidecode_text(collection[pid])

    # Tokenize
    q_toks = q_tokenizer.tokenize([query], add_special_tokens=True)[0]
    d_toks = d_tokenizer.tokenize([doc], add_special_tokens=True)[0]

    # # Remove doc toks in skiplists
    # skiplist = list(key for key in retriever.searcher.checkpoint.skiplist.keys() if not isint(key))
    # d_toks = [d_tok for d_tok in d_toks if d_tok not in skiplist]

    # Compute token scores
    result: RetrievalResult = retriever.calculate_score_by_text_batch(
        queries=[query], doc_texts=[doc]
    )[0]

    print(f"Query: {query}")
    print(f"Document (score: {result.score}):\n{doc}\n")
    examine_token_scores(q_toks, d_toks, result.token_scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
