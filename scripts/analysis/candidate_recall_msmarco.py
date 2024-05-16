import logging
import os
from typing import *

from beir.retrieval.custom_metrics import mrr
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

from model.late_encoder import ColBERTRetriever, RetrievalResult
from scripts.evaluate.utils import qrels_to_beir_format, results_to_beir_format
from scripts.utils import read_qrels, read_queries

# Data path
DATASET_DIR = "/root/ColBERT/data"
collection_path = os.path.join(DATASET_DIR, "msmarco/collection.tsv")
dev_query_path = os.path.join(DATASET_DIR, "msmarco/queries.dev.tsv")
dev_qrels_path = os.path.join(DATASET_DIR, "msmarco/qrels.dev.tsv")

# Retriever path
ROOT = "/root/ColBERT/experiments/"
EXPERIMENT = "msmarco"
# INDEX = "msmarco.no_unused_token.nbits=2"
# INDEX = "msmarco.my_hard.nbits=2"
# INDEX = "msmarco.distillation2.nbits=2"
INDEX = "msmarco.nbits=2"
SKIP_PADDING = False

logger = logging.getLogger("Evaluate")


def main():
    """Evaluate the retrieval performance of ColBERT."""
    # Read queries
    logger.info(f"Reading data...")
    query_dict = read_queries(dev_query_path)
    qrels = read_qrels(dev_qrels_path)
    # qrels = qrels[:7000]

    # Initialize retriever
    logger.info(f"Initializing retriever...")
    retriever = ColBERTRetriever(
        root=ROOT, index=INDEX, experiment=EXPERIMENT, skip_padding=SKIP_PADDING
    )

    # Get queries without duplicates
    qids = []
    qids_set = set()
    queries = []
    for qid, pid in tqdm(qrels):
        if qid not in qids_set:
            qids_set.add(qid)
            qids.append(qid)
            queries.append(query_dict[qid])

    # Search for top-k passages for each query
    logger.info(f"Searching top-k passages for {len(queries)} query...")
    topk_passages_list: List[List[RetrievalResult]] = (
        retriever.retrieve_candidates_batch(queries)
    )
    assert len(topk_passages_list) == len(
        qids
    ), f"Different number of queries and topk passages: {len(topk_passages_list)} vs {len(qids)}"
    topk_passages_dict: Dict[str, List[RetrievalResult]] = {
        qid: topk_passages for qid, topk_passages in zip(qids, topk_passages_list)
    }

    # Get document ids
    for key, results in topk_passages_dict.items():
        topk_passages_dict[key] = [result.doc.id for result in results]

    # Evaluate the retrieval performance
    logger.info(f"Evaluating...")
    top_5_recall = []
    top_10_recall = []
    top_50_recall = []
    top_100_recall = []
    top_all_recall = []
    # top_1000_recall = []
    for qid, positive_pid in tqdm(qrels):
        # Get top-k negative passages
        topk_passages = topk_passages_dict[qid]
        topk_passages = [str(pid) for pid in topk_passages]
        # Evaluate recall @5, @10, @50, @100
        top_5_recall.append(1 if positive_pid in topk_passages[:5] else 0)
        top_10_recall.append(1 if positive_pid in topk_passages[:10] else 0)
        top_50_recall.append(1 if positive_pid in topk_passages[:50] else 0)
        top_100_recall.append(1 if positive_pid in topk_passages[:100] else 0)
        top_all_recall.append(1 if positive_pid in topk_passages else 0)
        # top_1000_recall.append(1 if positive_pid in topk_passages[:1000] else 0)

    # Print the results
    logger.info(f"Total number of queries: {len(qrels)}")
    logger.info(f"Recall@5: {sum(top_5_recall)/len(top_5_recall)}")
    logger.info(f"Recall@10: {sum(top_10_recall)/len(top_10_recall)}")
    logger.info(f"Recall@50: {sum(top_50_recall)/len(top_50_recall)}")
    logger.info(f"Recall@100: {sum(top_100_recall)/len(top_100_recall)}")
    logger.info(f"Recall@all: {sum(top_all_recall)/len(top_all_recall)}")

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
