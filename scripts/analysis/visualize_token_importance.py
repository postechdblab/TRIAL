import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import torch
import tqdm
from omegaconf import DictConfig

from eagle.model import LightningNewModel
from eagle.phrase.pos import POSParser
from eagle.tokenization.tokenizer import Tokenizer
from scripts.inference import single_inference

logger = logging.getLogger("VisualizeTokenImportance")


def average_numbers_in_list(numbers: List[float]) -> float:
    return sum(numbers) / len(numbers)


def average_list_items_in_dict(dict: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: average_numbers_in_list(value) for key, value in dict.items()}


def get_pos_tags(sentences: List[str], pos_parser: POSParser) -> List[str]:
    query_pos_results = pos_parser(
        list(" ".join(sentences) for sentences in sentences.values()),
        max_tok_len=pos_parser.tokenizer.cfg.max_len,
    )
    return query_pos_results


def load_dataset(
    cfg: DictConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, cfg.dataset.val.data_file
    )
    corpus_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, cfg.dataset.corpus_file
    )
    query_file_path = os.path.join(
        cfg.dataset.dir_path, cfg.dataset.name, cfg.dataset.query_file
    )
    logger.info(f"Loading dataset from {dev_file_path}")
    dev_data = file_utils.read_jsonl_file(dev_file_path)
    logger.info(f"Loading dataset from {corpus_file_path}")
    corpus_data: List[Dict[str, Any]] = file_utils.read_jsonl_file(corpus_file_path)
    corpus_data: Dict[str, Dict[str, Any]] = {
        datum["_id"]: datum for datum in corpus_data
    }
    logger.info(f"Loading dataset from {query_file_path}")
    query_data: List[Dict[str, Any]] = file_utils.read_jsonl_file(query_file_path)
    query_data: Dict[str, Dict[str, Any]] = {
        int(datum["_id"]): datum for datum in query_data
    }

    return dev_data, corpus_data, query_data


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Get the output file path
    output_dir_path = os.path.join("/root/EAGLE/analysis", cfg.dataset.name)
    output_file_path = os.path.join(
        output_dir_path,
        f"token_importance_{cfg._global.tag}.json",
    )

    # Load trained model
    assert cfg.model.ckpt_path, "Please provide the path to the checkpoint"
    model = LightningNewModel.load_from_checkpoint(checkpoint_path=cfg.model.ckpt_path)
    model.eval()
    q_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.query, model_name=cfg.model.backbone_name
    )
    d_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.document, model_name=cfg.model.backbone_name
    )
    # Initialize the POS parser
    pos_parser = POSParser(tokenizer=q_tokenizer)

    # Load the dataset
    dev_data, corpus_data, query_data = load_dataset(cfg=cfg)

    # Get the texts
    sample_num = 100
    scores_by_pos_tag: Dict[str, List[float]] = {}
    weights_by_pos_tag: Dict[str, List[float]] = {}
    accumulated_scores: List[List[float]] = []
    accumulated_weights: List[List[float]] = []
    accumulated_tokens: List[List[str]] = []
    for dev_datum in tqdm.tqdm(dev_data[:sample_num], desc="Processing"):
        qid = dev_datum[0]
        pos_pid = dev_datum[1]
        neg_pids = dev_datum[30:40]
        q_text: List[str] = query_data[qid]["text"]
        d_text: List[str] = corpus_data[pos_pid]["text"]
        neg_d_texts: List[str] = [
            " ".join(corpus_data[pid]["text"]) for pid in neg_pids
        ]
        q_text = " ".join(q_text)
        d_text = " ".join(d_text)

        # Debugging: Get neg document scores
        for neg_d_text in neg_d_texts:
            neg_d_tok_scores, neg_d_tok_weights = single_inference(
                model=model,
                q_text=q_text,
                d_text=neg_d_text,
                q_tokenizer=q_tokenizer,
                d_tokenizer=d_tokenizer,
            )

        # Get pos tags
        pos_tags: Tuple[List[str], List[str], List[Tuple[int, int]]] = pos_parser(
            q_text, max_tok_len=pos_parser.tokenizer.cfg.max_len
        )[0]
        max_q_tok_scores, q_weights = single_inference(
            model=model,
            q_text=q_text,
            d_text=d_text,
            q_tokenizer=q_tokenizer,
            d_tokenizer=d_tokenizer,
        )
        all_scores: List[float] = []
        all_tokens: List[str] = []
        all_pos_tags: List[str] = []
        all_weights: List[float] = []

        # Get the CLS and QXQ scores
        cls_score = max_q_tok_scores[0]
        qxq_score = max_q_tok_scores[1]
        # Append the CLS and QXQ scores
        all_scores.append(cls_score)
        all_tokens.append("[CLS]")
        all_pos_tags.append("[CLS]")
        all_weights.append(q_weights[0])
        all_scores.append(qxq_score)
        all_tokens.append("[QXQ]")
        all_pos_tags.append("[QXQ]")
        all_weights.append(q_weights[1])

        # Get the text token scores
        for i in range(len(pos_tags[2])):
            tok_indices = pos_tags[2][i]
            scores: List[float] = max_q_tok_scores[tok_indices[0] : tok_indices[1]]
            weights: List[float] = q_weights[tok_indices[0] : tok_indices[1]]
            tok = pos_tags[0][i]
            pos_tag = pos_tags[1][i]
            # Append the token and its score
            all_scores.append(sum(scores) / len(scores))
            all_tokens.append(tok)
            all_pos_tags.append(pos_tag)
            all_weights.append(sum(weights) / len(weights))
        # Get the [SEP] score
        sep_score = max_q_tok_scores[-1]
        # Append the [SEP] score
        all_scores.append(sep_score)
        all_tokens.append("[SEP]")
        all_pos_tags.append("[SEP]")
        all_weights.append(q_weights[-1])
        # Print the results
        print(f"Query: {q_text}")
        print(f"Document: {d_text}")
        print(f"Scores: {all_scores}")
        print(f"Tokens: {all_tokens}")
        print(f"POS Tags: {all_pos_tags}")
        print(f"Weights: {all_weights}")
        # Append the scores to the dictionary
        for i in range(len(all_pos_tags)):
            pos_tag = all_pos_tags[i]
            score = all_scores[i]
            weight = all_weights[i]
            if pos_tag not in scores_by_pos_tag:
                scores_by_pos_tag[pos_tag] = []
                weights_by_pos_tag[pos_tag] = []
            scores_by_pos_tag[pos_tag].append(score)
            weights_by_pos_tag[pos_tag].append(weight)
        # Accumulate the scores, weights, and tokens
        accumulated_scores.append(all_scores)
        accumulated_weights.append(all_weights)
        accumulated_tokens.append(all_tokens)
    # Average the scores and weights
    scores_by_pos_tag: Dict[str, float] = average_list_items_in_dict(scores_by_pos_tag)
    weights_by_pos_tag: Dict[str, float] = average_list_items_in_dict(
        weights_by_pos_tag
    )
    # Save the results
    # Make the output directory if it doesn't exist
    logger.info(f"Saving the results to {output_file_path}")
    os.makedirs(output_dir_path, exist_ok=True)
    file_utils.write_json_file(
        {
            "tag": cfg._global.tag,
            "sample_num": sample_num,
            "scores": scores_by_pos_tag,
            "weights": weights_by_pos_tag,
        },
        output_file_path,
    )
    # Save the accumulated scores, weights, and tokens
    file_utils.write_json_file(
        {
            "scores": accumulated_scores,
            "weights": accumulated_weights,
            "tokens": accumulated_tokens,
        },
        os.path.join(output_dir_path, f"examples_{cfg._global.tag}.json"),
    )
    logger.info("Done")

    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
