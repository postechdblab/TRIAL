import logging
from typing import *

import torch

from scripts.utils import read_queries, read_qrels
from scripts.retrieval import Retriever


# Retriever path
ROOT = "/root/EAGLE/experiments/"
EXPERIMENT = "msmarco"
INDEX = "msmarco.nbits=2"
QUERY_PATH = "/root/EAGLE/data/msmarco/queries.dev.tsv"
QRELS_PATH = "/root/EAGLE/data/msmarco/qrels.dev.tsv"

logger = logging.getLogger("uniformityCheck")


def uniform_loss(x: torch.Tensor, t: int = 2) -> torch.Tensor:
    """
    bsz : batch size (number of positive pairs)
    d   : latent dim
    x   : Tensor, shape=[bsz, d]
          latents for one side of positive pairs
    y   : Tensor, shape=[bsz, d]
          latents for the other side of positive pairs
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def get_query_embeddings(query_data_path: str, retriever: Retriever) -> torch.Tensor:
    # Read in queries
    queries: Dict[str, str] = read_queries(query_data_path)
    # Sample 10000 queries
    queries = list(queries.values())[:1000]

    # Encode queries
    return retriever.get_encoded_queries(queries=queries).cuda()


def document_embeddings(qrels_path: str, retriever: Retriever) -> torch.Tensor:
    # Read in qrels
    qrels = read_qrels(qrels_path)
    # Get unique passage ids
    pids = []
    for qid, pid in qrels:
        if len(pids) == 250:
            break
        if pid not in pids:
            pids.append(pid)
    pids = [int(p) for p in pids]
    return retriever.get_encoded_documents(passage_ids=pids).float()


@torch.no_grad()
def main():
    # initialize retriever
    retriever = Retriever(root=ROOT, index=INDEX, experiment=EXPERIMENT)

    # Get query embeddings
    q_embeds = get_query_embeddings(query_data_path=QUERY_PATH, retriever=retriever)

    # Get document embeddings
    d_embeds = document_embeddings(qrels_path=QRELS_PATH, retriever=retriever)

    # Combine
    embeds = torch.cat((q_embeds, d_embeds), dim=0)

    # Compute uniform loss for query embeddings
    query_uniform_loss = uniform_loss(x=q_embeds)

    # Compute uniform loss for document embeddings
    doc_uniform_loss = uniform_loss(x=d_embeds)

    # Compute uniform loss for all query and document embeddings
    combined_uniform_loss = uniform_loss(x=embeds)

    logger.info(
        f"Query uniform loss: {query_uniform_loss} (on {q_embeds.shape[0]} tokens))"
    )
    logger.info(
        f"Document uniform loss: {doc_uniform_loss} (on {d_embeds.shape[0]} tokens))"
    )
    logger.info(
        f"Combined uniform loss: {combined_uniform_loss} (on {embeds.shape[0]} tokens))"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
