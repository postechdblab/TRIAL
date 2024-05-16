import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from model import RetrievalResult
from model.late_encoder import ColBERTRetriever
from scripts.utils import read_qrels, read_queries

# Retriever path
ROOT = "/root/ColBERT/experiments/"
# EXPERIMENT = "hotpotqa"
EXPERIMENT = "msmarco"
# INDEX = "hotpotqa.no_padding.nbits=2"
# INDEX = "hotpotqa.nbits=2"
INDEX = "msmarco.nbits=2"
IS_NEW_VERSION = False
IS_USE_NOUN = False
IS_USE_MIN_THRESHOLD = False
IS_USE_PHRASE_LEVEL = False
# Data path
DATASET_DIR = "/root/ColBERT/data"

collection_path = os.path.join(DATASET_DIR, "msmarco/collection.tsv")
dev_query_path = os.path.join(DATASET_DIR, "msmarco/queries.dev.tsv")
dev_qrels_path = os.path.join(DATASET_DIR, "msmarco/qrels.dev.tsv")

dev_path = os.path.join(DATASET_DIR, "hotpotqa/dev.json")

logger = logging.getLogger("Evaluate")


def main():
    """Evaluate the retrieval performance of ColBERT."""
    # Read queries
    logger.info(f"Reading data...")
    is_hotpotqa = False
    if is_hotpotqa:
        dev_data = file_utils.read_json_file(dev_path)
        # Filter noanwer
        dev_data = [
            item
            for item in dev_data
            if len(item["answers"]) > 0 and item["answers"][0].lower() != "no answer"
        ]
        # Filter only bridge type
        dev_data = [item for item in dev_data if item["type"].lower() == "comparison"]
        # dev_data = dev_data[:20]
    else:
        query_dict = read_queries(dev_query_path)
        qrels = read_qrels(dev_qrels_path)
        qrels = qrels[:10000]

    # Initialize retriever
    logger.info(f"Initializing retriever...")
    retriever = ColBERTRetriever(
        root=ROOT,
        index=INDEX,
        experiment=EXPERIMENT,
        is_use_phrase_level=IS_USE_PHRASE_LEVEL,
        is_use_min_threshold=IS_USE_MIN_THRESHOLD,
    )

    # Get queries without duplicates
    queries = []
    if is_hotpotqa:
        qids = []
        qrels = []
        for qid, item in enumerate(tqdm(dev_data)):
            qids.append(item["id"])
            queries.append(item["question"])
            # Get positive passage id
            positive_titles = list(set([t[0] for t in item["supporting_facts"]]))
            # Convert title to passage id
            positive_pids = [retriever.titles.index(title) for title in positive_titles]
            qrels.append(positive_pids)
    else:
        qids = []
        qids_set = set()
        for qid, pid in tqdm(qrels):
            if qid not in qids_set:
                qids_set.add(qid)
                qids.append(qid)
                queries.append(query_dict[qid])

    # Search for top-k passages for each query
    logger.info(f"Searching top-k passages for {len(queries)} query...")
    topk_passages_list: List[List[RetrievalResult]] = retriever.retrieve_batch(
        queries=queries
    )
    assert len(topk_passages_list) == len(
        qids
    ), f"Different number of queries and topk passages: {len(topk_passages_list)} vs {len(qids)}"
    topk_passages_dict = {
        qid: topk_docs for qid, topk_docs in zip(qids, topk_passages_list)
    }

    # Find the density
    only_gold = False
    all_token_scores = []
    for qid, positive_pid in tqdm(qrels):
        positive_pid = int(positive_pid)
        top_results = topk_passages_dict[qid]
        if only_gold:
            # Consider postiive document only
            results = [
                result for result in top_results if result.doc.id == positive_pid
            ]
        else:
            # Consider negative document only
            results = [
                result for result in top_results if result.doc.id != positive_pid
            ]
        # Pass if there is no result
        if not results:
            continue
        doc_texts = [result.doc.text for result in results]
        results, q_tokens, d_tokens = retriever.searcher.checkpoint.lazy_rank(
            queries=[query_dict[qid]] * len(doc_texts),
            docs=doc_texts,
            return_tokens=True,
        )
        results_max = torch.max(results, dim=-1)[0]
        all_token_scores.append(results_max)
    all_token_scores = torch.cat(all_token_scores, dim=0).reshape(-1).tolist()

    # Creating a density histogram with 10 bins
    plt.hist(all_token_scores, bins=10)

    # Adding labels and title for clarity
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Density Histogram of Random Data Between 0 and 1")

    # Displaying the histogram
    positive_or_negative = "positive" if only_gold else "negative"
    is_hotptoqa = "hotpotqa" if is_hotpotqa else "msmarco"
    file_name = f"{is_hotptoqa}_{positive_or_negative}_tmp.png"
    plt.savefig(file_name)
    logger.info(f"Saved density histogram to {file_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
