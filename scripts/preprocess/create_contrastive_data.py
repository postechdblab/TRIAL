import logging
import os
from typing import *

import hkkang_utils.file as file_utils
from tqdm import tqdm

from model.late_encoder import ColBERTRetriever, RetrievalResult
from scripts.utils import read_qrels, read_queries

# Data path
DATASET_DIR = "/root/ColBERT/data"
collection_path = os.path.join(DATASET_DIR, "msmarco_old/collection.tsv")
train_query_path = os.path.join(DATASET_DIR, "msmarco_old/queries.train.tsv")
train_qrels_path = os.path.join(DATASET_DIR, "msmarco_old/qrels.train.tsv")

# Retriever path
ROOT = "/root/ColBERT/experiments/"
EXPERIMENT = "msmarco_unidecode"
INDEX = "msmarco.colbertv2.0.nbits=2"

# Output file name
OUTPUT_FILE_PATH = "msmarco_old/train_data3.jsonl"

logger = logging.getLogger("GenerateData")


def main() -> None:
    """Generate training data for the model."""
    # Read queries
    logger.info(f"Reading data...")
    query_dict = read_queries(train_query_path)
    qrels = read_qrels(train_qrels_path)
    # qrels = qrels[:100]

    # Initialize retriever
    logger.info(f"Initializing retriever...")
    retriever = ColBERTRetriever(root=ROOT, index=INDEX, experiment=EXPERIMENT)

    # Get queries without duplicates
    qids = []
    queries = []
    qid_set = set()
    for qid, pid in tqdm(qrels):
        if qid not in qid_set:
            qids.append(qid)
            qid_set.add(qid)
            queries.append(query_dict[qid])

    # Search for top-k passages for each query
    logger.info(f"Searching top-k passages for {len(queries)} query...")
    all_results: List[List[RetrievalResult]] = retriever.retrieve_batch(
        queries, topk=256
    )
    assert len(all_results) == len(
        qids
    ), f"Different number of queries and topk passages: {len(all_results)} vs {len(qids)}"
    topk_passages_dict = {qid: result for qid, result in zip(qids, all_results)}

    # Remove positive passages from top-k passages (to leave negative passages only)
    logger.info(f"Removing positive passages from top-k passages...")
    for qid, pid in tqdm(qrels):
        if pid in topk_passages_dict[qid]:
            topk_passages_dict[qid].remove(pid)

    # Generate training data
    logger.info(f"Generating training data...")
    data_list = []
    for qid, positive_pid in tqdm(qrels):
        # Get top-k negative passages
        results: List[RetrievalResult] = topk_passages_dict[qid]
        negative_passages = [result.doc.id for result in results]
        # Aggregate
        ids = [qid, positive_pid, *negative_passages]
        data_list.append([int(item) for item in ids])

    # Save data
    logger.info(f"Saving {len(data_list)} data... to {DATASET_DIR}")
    data_path = os.path.join(DATASET_DIR, OUTPUT_FILE_PATH)
    file_utils.write_jsonl_file(data_list, data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
