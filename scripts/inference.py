import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
from typing import *

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from eagle.model import LightningNewModel
from eagle.tokenization.tokenizer import Tokenizer
from eagle.utils import add_global_configs, set_random_seed
from scripts.utils import (
    check_argument,
    format_preprocessed_data_as_batch,
    preprocess,
    pretty_print_tokens_with_their_indices,
)

logger = logging.getLogger("Evaluate")


@torch.no_grad()
def single_inference(
    model: LightningNewModel,
    q_text: str,
    d_text: str,
    q_tokenizer: Tokenizer,
    d_tokenizer: Tokenizer,
) -> None:
    # Prepare the input
    preprocessed_query = preprocess([q_text], tokenizer=q_tokenizer)
    preprocessed_document = preprocess([d_text], tokenizer=d_tokenizer)

    # Create batch input
    preprocessed_batch = format_preprocessed_data_as_batch(
        preprocessed_query=preprocessed_query,
        preprocessed_document=preprocessed_document,
        model_device=model.device,
    )

    # Forward the model
    results = model.model(**preprocessed_batch)

    # Move the result tensors to the CPU
    results = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in results.items()
    }

    # Get the query-doc token similarity scores
    if model.model.cfg.name == "colbert":
        qd_sims = results["intra_qd_scores"].squeeze().transpose(0, 1)
        decoded_d_tokens = [
            d_tokenizer.decode(item) for item in preprocessed_document["tok_ids"][0]
        ]
        d_weights = None
        q_weights = None
    elif model.model.cfg.name == "eagle":
        qd_sims = results["intra_qd_outer_scores"].squeeze().transpose(0, 1)
        # Concate tok, phrase, sent
        # Get sentence indices
        doc_toks = [
            d_tokenizer.decode(item) for item in preprocessed_document["tok_ids"][0]
        ]
        sent_start_toks = [
            doc_toks[idx] for idx in preprocessed_document["sent_start_indices"][0]
        ]
        phrases = [
            "_".join(doc_toks[s:e])
            for s, e in preprocessed_document["phrase_ranges"][0]
        ]
        # decoded_d_tokens = sent_start_toks + phrases + doc_toks
        decoded_d_tokens = doc_toks
        d_weights = results["intra_selected_d_weights"].squeeze()
        q_weights = results["intra_q_weights"].squeeze()
    else:
        raise ValueError(f"Invalid model name: {model.model.cfg.name}")

    # Find the max scores for each token
    if q_weights is None:
        max_scores, max_indices = qd_sims.max(1)
    else:
        qd_scores = qd_sims * q_weights.unsqueeze(1) * d_weights.unsqueeze(1)
        max_scores, max_indices = qd_scores.max(1)
    # Find the total score of the query-document
    total_score = max_scores.sum()

    # Analyze
    # Show the max document token and score for each query token
    logger.info(f"Total score: {total_score}")
    for i, (max_score, max_idx) in enumerate(zip(max_scores, max_indices)):
        query_token = q_tokenizer.decode(preprocessed_query["tok_ids"][0][i])
        doc_token = decoded_d_tokens[max_idx]
        if q_weights is None:
            logger.info(
                f"Q Token {i:2d}: {query_token:<10}\t->\tD token {max_idx:2d}: {doc_token:<10}\t(Score: {max_score:.5f})"
            )
        else:
            q_weight = q_weights[i]
            d_weight = d_weights[i]
            sim_score = qd_sims[i, max_idx]
            logger.info(
                f"Q Token {i:2d}: {query_token:<10}\t->\tD token {max_idx:2d}: {doc_token:<10}\t(Score: {max_score:.3f}) (sim:{sim_score:.3f}) (q_w:{q_weight:.3f}) (d_w:{d_weight:.3f})"
            )

    # Print the document tokens with their indices
    decoded_q_tokens = [
        q_tokenizer.decode([token_id]) for token_id in preprocessed_query["tok_ids"][0]
    ]
    print("Query tokens:")
    pretty_print_tokens_with_their_indices(
        decoded_tokens=decoded_q_tokens, max_tokens_per_line=20
    )
    print("\nDocument tokens:")
    pretty_print_tokens_with_their_indices(
        decoded_tokens=decoded_d_tokens, max_tokens_per_line=20
    )
    print("")


def inference(cfg: DictConfig, ckpt_path: str, is_analyze: bool = True) -> None:
    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    q_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.query, model_name=cfg.model.backbone_name
    )
    d_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.document, model_name=cfg.model.backbone_name
    )

    # Get input
    while True:
        # query_text = "who is the scientist"
        # document_text = "who is the scientist"
        query_text = input("Enter the query: ")
        document_text = input("Enter the document: ")
        # Check if the query is empty
        if query_text == "q":
            logger.info("Exit the program")
            break

        single_inference(
            model=model,
            q_text=[query_text],
            d_text=[document_text],
            q_tokenizer=q_tokenizer,
            d_tokenizer=d_tokenizer,
        )

    return None


def check_arguments(cfg: DictConfig) -> DictConfig:
    check_argument(
        cfg.args,
        name="use_slack",
        arg_type=bool,
        help="Whether to use slack notification",
    )
    check_argument(
        cfg.args, name="ckpt_path", arg_type=str, help="Path to the checkpoint"
    )
    return cfg.args


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg: DictConfig = add_global_configs(cfg, exclude_keys=["args"])

    # Set random seeds
    L.seed_everything(cfg._global.seed, workers=True)
    args = check_arguments(cfg)

    with torch.no_grad():
        inference(cfg, ckpt_path=args.ckpt_path)

        logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    set_random_seed()
    main()
