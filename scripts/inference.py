import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
from typing import *

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from eagle.dataset.utils import get_att_mask, get_mask
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
from eagle.tokenization.utils import combine_splitted_tok_ids
from eagle.utils import add_global_configs, set_random_seed
from scripts.utils import check_argument, pretty_print_tokens_with_their_indices

logger = logging.getLogger("Evaluate")


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
    scatter_indices = torch.tensor(scatter_indices, dtype=torch.long)
    # Convert list to tensor
    tok_ids_tensor = torch.tensor(tok_ids)
    # Create token mask
    tok_mask = get_mask(tok_ids_tensor, skip_ids=tokenizer.skip_tok_ids)
    tok_att_mask = get_att_mask(tok_ids_tensor, skip_ids=[0])
    phrase_mask = torch.zeros(len(phrase_ranges), dtype=torch.bool).float()
    sent_mask = torch.zeros(len(sent_start_indices), dtype=torch.bool).float()

    return {
        "tok_ids": tok_ids_tensor,
        "tok_att_mask": tok_att_mask,
        "tok_mask": tok_mask,
        "sent_start_indices": sent_start_indices,
        "phrase_scatter_indices": scatter_indices,
        "phrase_mask": phrase_mask,
        "sent_mask": sent_mask,
        "phrase_ranges": phrase_ranges,
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
            "q_sent_start_indices": [[preprocessed_query["sent_start_indices"]]],
            "doc_sent_start_indices": [[preprocessed_document["sent_start_indices"]]],
            "q_phrase_scatter_indices": preprocessed_query[
                "phrase_scatter_indices"
            ].unsqueeze(0),
            "doc_phrase_scatter_indices": preprocessed_document[
                "phrase_scatter_indices"
            ].unsqueeze(0),
            "q_phrase_mask": preprocessed_query["phrase_mask"].unsqueeze(0),
            "doc_phrase_mask": preprocessed_document["phrase_mask"]
            .unsqueeze(0)
            .unsqueeze(0),
            "q_sent_mask": preprocessed_query["sent_mask"].unsqueeze(0),
            "doc_sent_mask": preprocessed_document["sent_mask"]
            .unsqueeze(0)
            .unsqueeze(0),
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
        if "intra_qd_scores" in results:
            qd_scores = results["intra_qd_scores"].squeeze().transpose(0, 1)
            decoded_d_tokens = [
                d_tokenizer.decode(item) for item in preprocessed_document["tok_ids"]
            ]
        else:
            qd_scores = results["intra_qd_outer_scores"].squeeze().transpose(0, 1)
            # Concate tok, phrase, sent
            # Get sentence indices
            doc_toks = [
                d_tokenizer.decode(item) for item in preprocessed_document["tok_ids"]
            ]
            sent_start_toks = [
                doc_toks[idx] for idx in preprocessed_document["sent_start_indices"]
            ]
            phrases = [
                "_".join(doc_toks[s:e])
                for s, e in preprocessed_document["phrase_ranges"]
            ]
            decoded_d_tokens = sent_start_toks + phrases + doc_toks

        # Find the max scores for each token
        max_scores, max_indices = qd_scores.max(1)

        # Find the total score of the query-document
        total_score = max_scores.sum()

        # Analyze
        # Show the max document token and score for each query token
        logger.info(f"Total score: {total_score}")
        for i, (max_score, max_idx) in enumerate(zip(max_scores, max_indices)):
            query_token = q_tokenizer.decode([preprocessed_query["tok_ids"][i]])
            doc_token = decoded_d_tokens[max_idx]
            logger.info(
                f"Q Token {i:2d}: {query_token:<10}\t->\tD token {max_idx:2d}: {doc_token:<10}\t(Score: {max_score:.5f})"
            )

        # Print the document tokens with their indices
        decoded_q_tokens = [
            q_tokenizer.decode([token_id]) for token_id in preprocessed_query["tok_ids"]
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
