import logging
import os
from typing import *

import hkkang_utils.slack as slack_utils
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from eagle.dataset import ContrastiveDataModule, InferenceDataModule
from eagle.dataset.utils import combine_splitted_tok_ids, get_att_mask, get_mask
from eagle.model import LightningNewModel
from eagle.model.batch.utils import (
    convert_range_to_scatter,
    cut_off_phrase_ranges_by_max_len,
)
from eagle.phrase import (
    PhraseExtractor,
    combined_phrase_ranges_into_one_sentence,
    fix_bad_index_ranges,
)
from eagle.tokenization import Sentencizer, Tokenizer
from eagle.utils import add_global_configs, set_random_seed
from scripts.utils import check_argument, pretty_print_tokens_with_their_indices

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("Run")


def preprocess(text: str, tokenizer: Tokenizer) -> Any:
    # Split the text into sentences
    sentences: List[str] = Sentencizer()(text)
    # Tokenize the text
    tokenized_sentences = tokenizer(sentences)["input_ids"]
    # Extract the phrases
    extractor = PhraseExtractor(tokenizer=tokenizer)
    # Extract phrases and combine them as a single sentence
    phrase_ranges_per_sent: List[List[Tuple[int, int]]] = extractor(
        texts=sentences,
        tok_ids_list=tokenized_sentences,
        to_token_indices=True,
    )
    phrase_ranges = combined_phrase_ranges_into_one_sentence(
        [fix_bad_index_ranges(item) for item in phrase_ranges_per_sent]
    )

    # Preprocess as the model input
    # Combine the splitted sentences
    tok_ids, sent_start_indices = combine_splitted_tok_ids(tokenized_sentences)
    # Cut-off by max length
    tok_ids = tokenizer.cutoff_by_max_len(tok_ids)
    phrase_ranges = cut_off_phrase_ranges_by_max_len(
        phrase_ranges, tokenizer.cfg.max_len
    )
    # Get scatter indices for phrases
    scatter_indices: List[int] = convert_range_to_scatter(phrase_ranges)
    # # Cut off phrase scatter indices if it exceeds the maximum length
    # scatter_indices = tokenizer.cutoff_by_max_len(
    #     scatter_indices, maintain_special_tokens=False
    # )
    # Convert list to tensor
    tok_ids_tensor = torch.tensor(tok_ids)
    # Create token mask
    tok_mask = get_mask(tok_ids_tensor, skip_ids=tokenizer.skip_tok_ids)
    tok_att_mask = get_att_mask(tok_ids_tensor, skip_ids=[0])

    return {
        "tok_ids": tok_ids_tensor,
        "tok_att_mask": tok_att_mask,
        "tok_mask": tok_mask,
        "sent_start_indices": sent_start_indices,
        "phrase_scatter_indices": scatter_indices,
    }


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

        # Prepare the input
        preprocessed_query = preprocess(query_text, tokenizer=q_tokenizer)
        preprocessed_document = preprocess(document_text, tokenizer=d_tokenizer)

        # Create batch input
        preprocessed_batch = {
            "q_tok_ids": preprocessed_query["tok_ids"].unsqueeze(0),
            "q_tok_att_mask": preprocessed_query["tok_att_mask"].unsqueeze(0),
            "q_tok_mask": preprocessed_query["tok_mask"].unsqueeze(0),
            "doc_tok_ids": preprocessed_document["tok_ids"].unsqueeze(0).unsqueeze(0),
            "doc_tok_att_mask": preprocessed_document["tok_att_mask"]
            .unsqueeze(0)
            .unsqueeze(0),
            "doc_tok_mask": preprocessed_document["tok_mask"].unsqueeze(0).unsqueeze(0),
            "labels": None,
            "distillation_scores": None,
            "pos_doc_ids": None,
            "is_analyze": True,
        }
        # Move the tensors to the device same as the model
        preprocessed_batch = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed_batch.items()
        }

        # Forward the model
        results = model.model(**preprocessed_batch)

        # Move the result tensors to the CPU
        results = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in results.items()
        }

        # Get the query-doc token similarity scores
        qd_scores = results["intra_qd_scores"].squeeze().transpose(0, 1)

        # Find the max scores for each token
        max_scores, max_indices = qd_scores.max(1)

        # Find the total score of the query-document
        total_score = max_scores.sum()

        # Analyze
        # Show the max document token and score for each query token
        logger.info(f"Total score: {total_score}")
        for i, (max_score, max_idx) in enumerate(zip(max_scores, max_indices)):
            query_token = q_tokenizer.decode([preprocessed_query["tok_ids"][i]])
            doc_token = d_tokenizer.decode([preprocessed_document["tok_ids"][max_idx]])
            logger.info(
                f"Q Token {i:2d}: {query_token:<10}\t->\tD token {max_idx:2d}: {doc_token:<10}\t(Score: {max_score:.5f})"
            )

        # Print the document tokens with their indices
        decoded_q_tokens = [
            q_tokenizer.decode([token_id]) for token_id in preprocessed_query["tok_ids"]
        ]
        decoded_d_tokens = [
            d_tokenizer.decode([token_id])
            for token_id in preprocessed_document["tok_ids"]
        ]
        print("Query tokens:")
        pretty_print_tokens_with_their_indices(
            decoded_tokens=decoded_q_tokens, max_tokens_per_line=20
        )
        print("Document tokens:")
        pretty_print_tokens_with_their_indices(
            decoded_tokens=decoded_d_tokens, max_tokens_per_line=20
        )
        print("")

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
    remove_model_prefix_key_from_saved_dict(ckpt_path=ckpt_path)
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
    # remove_model_prefix_key_from_saved_dict(ckpt_path=ckpt_path)
    trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    return None


def remove_model_prefix_key_from_saved_dict(ckpt_path: str) -> None:
    logger.info(f"Loding the checkpoint from {ckpt_path}")
    tmp = torch.load(ckpt_path)
    logger.info(f"Removing the prefix from the model state_dict")
    tmp["state_dict"] = {
        k.replace("._orig_mod.", "."): v for k, v in tmp["state_dict"].items()
    }
    logger.info(f"Saving the modified checkpoint to {ckpt_path}")
    torch.save(tmp, ckpt_path)
    return None


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
        choices=["inference", "evaluate_retrieval", "evaluate_reranking"],
        is_requried=True,
        help="mode should be 'inference' ,'evaluate_retrieval', or 'evaluate_reranking'",
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
