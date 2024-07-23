import logging
import os
from typing import *

import hkkang_utils.slack as slack_utils
import hydra
import lightning as L
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

from eagle.dataset import ContrastiveDataModule, InferenceDataModule
from eagle.dataset.utils import (
    add_doc_ranges_and_mask,
    add_query_ranges_and_mask,
    collate_fn,
)
from eagle.index.indexer import multi_process_indexing
from eagle.model import LightningNewModel
from eagle.phrase.extraction import PhraseExtractor
from eagle.tokenizer import DTokenizer, QTokenizer
from eagle.utils import add_global_configs, set_random_seed
from scripts.utils import check_argument, join_word

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("Run")


def inference(cfg: DictConfig, ckpt_path: str, is_analyze: bool = True) -> None:
    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    q_tokenizer = QTokenizer(cfg=cfg.q_tokenizer)
    d_tokenizer = DTokenizer(cfg=cfg.d_tokenizer)

    # Get phrase extractor
    phrase_extractor = PhraseExtractor(tokenizer=q_tokenizer)

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

        # Preprocess input query and document text
        input_dict = {
            "q_texts": query_text,
            "pos_doc_text_list": document_text,
            "neg_doc_texts_list": [],
        }
        raise NotImplementedError("Fix below code")
        preprocessed_batch = preprocess(
            input_dict,
            q_tokenizer=q_tokenizer,
            d_tokenizer=d_tokenizer,
            is_eval=True,
            unbatch=True,
            is_compress=False,
        )

        # Extract word indices
        q_word_indices = []
        d_word_indices = [[]]
        # Extract phrase indices
        q_phrase_indices = phrase_extractor(
            input_dict["q_texts"], max_len=cfg.q_tokenizer.max_len
        )[0]
        d_phrase_indices = phrase_extractor(
            input_dict["pos_doc_text_list"], max_len=cfg.d_tokenizer.max_len
        )

        # Add ranges and masks
        preprocessed_batch, q_ranges = add_query_ranges_and_mask(
            input_dict=preprocessed_batch,
            word_ranges=q_word_indices,
            phrase_ranges=q_phrase_indices,
            skip_ids=q_tokenizer.special_toks_ids,
            use_coarse_emb=model.model.is_use_multi_granularity,
            return_ranges=True,
        )
        preprocessed_batch, d_ranges = add_doc_ranges_and_mask(
            input_dict=preprocessed_batch,
            word_ranges=d_word_indices,
            phrase_ranges=d_phrase_indices,
            skip_ids=d_tokenizer.special_toks_ids + d_tokenizer.punctuations,
            use_coarse_emb=model.model.is_use_multi_granularity,
            return_ranges=True,
        )
        # Collate
        collated_batch = collate_fn([preprocessed_batch])

        log_dict, scores = model(**collated_batch, is_analyze=is_analyze)

        if is_analyze:
            # Get the query terms
            q_tokens = q_tokenizer.tokenizer.convert_ids_to_tokens(
                collated_batch["q_tok_ids"][0]
            )
            q_terms = [join_word(q_tokens, start, end) for (start, end) in q_ranges]
            q_weights = (
                [1 for _ in q_terms]
                if log_dict["q_weight"] is None
                else log_dict["q_weight"]
            )
            # Get the document terms
            d_tokens = d_tokenizer.tokenizer.convert_ids_to_tokens(
                collated_batch["doc_tok_ids"][0][0]
            )
            d_terms = [join_word(d_tokens, start, end) for (start, end) in d_ranges[0]]
            d_weights = (
                [1 for _ in d_terms]
                if log_dict["d_weight_intra"] is None
                else log_dict["d_weight_intra"]
            )
            # Show the document score
            logger.info(f"Relevance score: {scores.item()}")
            logger.info(f"Query: {query_text}")
            logger.info(f"Document: {document_text}")
            logger.info(f"Query {len(q_terms)} terms: {q_terms}")
            logger.info(f"Document {len(d_terms)} terms: {d_terms}")
            if log_dict["d_weight_intra"] is not None:
                # TODO: Print document weights
                raise NotImplementedError(
                    "Analysis on document weights is not implemented yet"
                )
            # Show detailed scores
            if log_dict["intra_qd_scores"] is not None:
                qd_scores: torch.Tensor = log_dict["intra_qd_scores"]
                # Find the max doc value and indices for each query term
                max_doc_values, max_doc_indices = torch.max(qd_scores, dim=1)
                # Print the mapping and value
                for i, (term, max_doc_value, max_doc_index) in enumerate(
                    zip(q_terms, max_doc_values, max_doc_indices)
                ):
                    suffix = "\n" if i == len(q_terms) - 1 else ""
                    logger.info(
                        f"Q Term ({i}th): {term}\t<-> {d_terms[max_doc_index.item()]} (Score: {max_doc_value.item():.4f}, d-idx: {max_doc_index.item()}){suffix}"
                    )
                # Print the score for each query term
                for i, (term, score, weight) in enumerate(
                    zip(q_terms, max_doc_values, q_weights)
                ):
                    suffix = "\n" if i == len(q_terms) - 1 else ""
                    logger.info(
                        f"Q Term ({i}th): {term}\t| Scores: {score:.4f} | Weight: {weight} | Weighted Score: {(score*weight):.4f}{suffix}"
                    )
        is_continue = input("Do you want to continue? (y/n): ")
        if is_continue == "n":
            break

    return None


def full_retrieval(cfg: DictConfig, ckpt_path: str, is_analyze: bool) -> None:
    # Load data module and model
    data_module = InferenceDataModule(cfg)

    # Load index
    index_dir_path = os.path.join(
        cfg.indexing.dir_path, cfg.dataset.name, cfg._global.tag
    )
    logger.info(f"Index directory path: {index_dir_path}")

    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel(cfg=cfg, index_dir_path=index_dir_path)

    trainer = L.Trainer(
        deterministic=True,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy="ddp",
    )
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    return None


def reranking(cfg: DictConfig, ckpt_path: str, is_analyze: bool) -> None:
    # Load data module and model
    data_module = ContrastiveDataModule(cfg, skip_train=True)

    # Load trained model
    assert ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel(cfg=cfg)
    trainer = L.Trainer(
        deterministic=True,
        accelerator="cuda",
        devices=torch.cuda.device_count(),
        strategy="ddp",
    )
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    return None


def indexing(cfg: DictConfig) -> None:
    # Get configs
    world_size = torch.cuda.device_count()

    mp.spawn(
        multi_process_indexing,
        args=(cfg, world_size),
        nprocs=world_size,
        join=True,
    )
    print("Done!")


def run(
    cfg: DictConfig,
    mode: str,
    is_analyze: bool,
    ckpt_path: str = None,
    use_slack: bool = False,
) -> None:
    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to train NewRetriever!",
        error_msg=f"Falied to train NewRetriever!",
        disable=not use_slack,
    ):
        with torch.no_grad():
            if mode == "inference":
                return inference(cfg, ckpt_path=ckpt_path, is_analyze=is_analyze)
            elif mode == "indexing":
                return indexing(cfg)
            elif mode == "evaluate_retrieval":
                return full_retrieval(cfg, ckpt_path=ckpt_path, is_analyze=is_analyze)
            elif mode == "evaluate_reranking":
                return reranking(cfg, ckpt_path=ckpt_path, is_analyze=is_analyze)
            raise ValueError(f"Invalid mode: {mode}")


def check_arguments(cfg: DictConfig) -> DictConfig:
    check_argument(
        cfg.args,
        name="mode",
        arg_type=str,
        choices=["inference", "indexing", "evaluate_retrieval", "evaluate_reranking"],
        is_requried=True,
        help="mode should be 'inference', 'indexing' ,'evaluate_retrieval', or 'evaluate_reranking'",
    )
    check_argument(
        cfg.args, name="is_analyze", arg_type=bool, help="Whether to analyze the model"
    )
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

    # Check arguments
    args = check_arguments(cfg)

    run(cfg, **args)
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    set_random_seed()
    main()
