import json
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.list as list_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.model import LightningNewModel
from eagle.phrase.pos import POSParser
from eagle.tokenization.tokenizers import Tokenizers
from scripts.utils import format_preprocessed_data_as_batch, preprocess

logger = logging.getLogger("POSStats")


def get_stats_for_msmarco(
    cfg: DictConfig, sample_num: int = 1000
) -> Dict[str, Tuple[float, int, Tuple[str, str, str]]]:
    # Check arguments
    assert "ckpt_path" in cfg
    ckpt_path = cfg.ckpt_path

    # Load model
    model = LightningNewModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()

    # Get the dataset name and path
    dir_path = os.path.join(cfg.dataset.dir_path, cfg.dataset.name)
    train_path = os.path.join(dir_path, cfg.dataset.val.data_file)
    query_path = os.path.join(dir_path, cfg.dataset.query_file)
    corpus_path = os.path.join(dir_path, cfg.dataset.corpus_file)

    # Load the dev dataset
    logger.info(f"Loading the {cfg.dataset.name} dataset from {dir_path} ...")
    dev_data = file_utils.read_json_file(train_path, auto_detect_extension=True)
    logger.info(f"Loaded {len(dev_data)} samples from {train_path}")
    # Sample the data
    dev_data = dev_data[:sample_num]
    dev_queries = set(item[0] for item in dev_data)
    dev_docs = set(list_utils.do_flatten_list([item[1:] for item in dev_data]))
    # Load the query
    logger.info(f"Loading the query dataset from {query_path} ...")
    query_data = file_utils.read_json_file(query_path, auto_detect_extension=True)
    sampled_query_data = {
        item["_id"]: item["text"]
        for item in query_data
        if int(item["_id"]) in dev_queries
    }
    logger.info(
        f"Loaded {len(query_data)} samples from {query_path} and sampled {len(sampled_query_data)} samples"
    )
    # Load the corpus
    logger.info(f"Loading the corpus dataset from {corpus_path} ...")
    corpus_data = file_utils.read_json_file(corpus_path, auto_detect_extension=True)
    sampled_corpus_data = {
        item["_id"]: item["text"] for item in corpus_data if item["_id"] in dev_docs
    }
    logger.info(
        f"Loaded {len(corpus_data)} samples from {corpus_path} and sampled {len(sampled_corpus_data)} samples"
    )

    # Initialize tokenizers and extractors in each process
    tokenizers = Tokenizers(
        q_cfg=cfg.tokenizers.query,
        d_cfg=cfg.tokenizers.document,
        model_name=cfg.model.backbone_name,
    )
    # Initialize the POS parser
    q_extractor = POSParser(tokenizer=tokenizers.q_tokenizer)
    d_extractor = POSParser(tokenizer=tokenizers.d_tokenizer)

    # Figure out the pos tags for the query and document
    logger.info(
        f"Finding the pos tags for the {cfg.dataset.name} query dataset ({len(sampled_query_data)} data)"
    )
    query_pos_results = q_extractor(
        list(" ".join(item) for item in sampled_query_data.values()),
        max_tok_len=tokenizers.q_tokenizer.cfg.max_len,
    )
    logger.info(f"Found the pos tags for {len(query_pos_results)} queries")
    logger.info(
        f"Finding the pos tags for the {cfg.dataset.name} document dataset ({len(sampled_corpus_data)} data)"
    )
    corpus_pos_results = d_extractor(
        list(" ".join(item) for item in sampled_corpus_data.values()),
        max_tok_len=tokenizers.d_tokenizer.cfg.max_len,
        show_progress=True,
    )
    logger.info(f"Found the pos tags for {len(corpus_pos_results)} documents")

    # Compute the scores for the query and document
    pos_scores_dic: Dict[str, List[float]] = {}
    pos_sample_words: Dict[str, List[str]] = {}
    for i in tqdm.tqdm(range(len(dev_data)), desc="Computing the scores"):
        # Get pos
        query_pos = query_pos_results[i]
        doc_pos = corpus_pos_results[i]
        # Get query
        query_id: int = dev_data[i][0]
        query_text: str = " ".join(sampled_query_data[str(query_id)])
        # Get document
        doc_ids: List[int] = dev_data[i][1:][:10]
        doc_texts: List[str] = [" ".join(sampled_corpus_data[item]) for item in doc_ids]
        # Preprocess
        preprocssed_query = preprocess(
            [query_text], tokenizer=tokenizers.q_tokenizer, extract_phrase=False
        )
        preprocessed_document = preprocess(
            doc_texts, tokenizer=tokenizers.d_tokenizer, extract_phrase=False
        )
        preprocessed_batch = format_preprocessed_data_as_batch(
            preprocessed_query=preprocssed_query,
            preprocessed_document=preprocessed_document,
            model_device=model.device,
        )

        # Forward the model
        results = model.model(**preprocessed_batch)
        qd_scores = results["intra_qd_scores"].transpose(-1, -2)
        q_max_scores = qd_scores.max(dim=-1).values

        # Compute the scores for each pos tags (use the top-1 document for the computed score)
        for j in range(len(query_pos[0])):
            word = query_pos[0][j]
            pos = query_pos[1][j]
            tok_ranges = query_pos[2][j]
            avg_score = (
                sum(q_max_scores[0][tok_ranges[0] : tok_ranges[1]])
                / (tok_ranges[1] - tok_ranges[0])
            ).item()
            pos_scores_dic.setdefault(pos, []).append(avg_score)
            pos_sample_words.setdefault(pos, []).append(word)

    # Return the results
    final_results: Dict[str, Tuple[float, int, Tuple[str, str, str]]] = {}
    for key, values in pos_scores_dic.items():
        avg_scores = sum(values) / len(values)
        cnt = len(values)
        # Print the results
        logger.info(f"POS: {key} - Avg Score: {avg_scores:.4f} - Count: {cnt}")
        logger.info(f"POS: {key} - Sample Words: {pos_sample_words[key][:5]}")
        final_results[key] = (avg_scores, cnt, pos_sample_words[key][:3])

    return final_results


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    all_stats = get_stats_for_msmarco(cfg=cfg)
    logger.info(json.dumps(all_stats, indent=4))
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
