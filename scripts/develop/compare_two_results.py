import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hydra
import tqdm
from omegaconf import DictConfig

from eagle.model import LightningNewModel
from eagle.tokenization.tokenizer import Tokenizer
from scripts.inference import single_inference

logger = logging.getLogger("CompareTwoResults")


def detailed_comparison_with_qid(
    model1: LightningNewModel,
    model2: LightningNewModel,
    q_text: str,
    d_text: str,
    q_tokenizer: Tokenizer,
    d_tokenizer: Tokenizer,
) -> None:
    # Inference
    logger.info(f"Model1: ")
    result1 = single_inference(
        model=model1,
        q_text=q_text,
        d_text=d_text,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
    )
    logger.info(f"Model2: ")
    result2 = single_inference(
        model=model2,
        q_text=q_text,
        d_text=d_text,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
    )
    return None


def detailed_comparison_with_qids(
    model1: LightningNewModel,
    model2: LightningNewModel,
    q_tokenizer: Tokenizer,
    d_tokenizer: Tokenizer,
    query_data: Dict[str, str],
    corpus_data: Dict[str, Dict[str, str]],
    qids_to_compare: List[str],
    results1: Dict[str, Tuple[List[float], List[int]]],
    results2: Dict[str, Tuple[List[float], List[int]]],
    gold_pids_dic: Dict[str, List[int]],
) -> None:
    for qid_to_compare in qids_to_compare:
        gold_pid: int = gold_pids_dic[qid_to_compare][0]
        # Get the query and document texts
        query_sentences: List[str] = query_data[qid_to_compare]
        # Get the top-ranked document for model 1
        top_ranked_pids1: List[int] = results1[qid_to_compare][1]
        top_ranked_pids2: List[int] = results2[qid_to_compare][1]

        doc_sentences: List[str] = corpus_data[gold_pid]["text"]
        doc_title: str = corpus_data[gold_pid]["title"]
        # Create query and doc text
        q_text = " ".join(query_sentences)
        d_text = " ".join(doc_sentences)
        if doc_title:
            d_text = doc_title + " | " + d_text
        detailed_comparison_with_qid(
            model1=model1,
            model2=model2,
            q_text=q_text,
            d_text=d_text,
            q_tokenizer=q_tokenizer,
            d_tokenizer=d_tokenizer,
        )
        stop = 1
    stop = 1
    return None


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configs
    eval_dir_path = cfg._global.eval_dir
    dataset_name = "beir-arguana"
    query_path: str = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.query_file
    )
    corpus_path: str = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.corpus_file
    )
    dev_file_path = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.val.data_file
    )
    model1 = "colbert"
    model2 = "eagle"
    tag1 = "colbert"
    tag2 = "eagle_weights"
    model1_ckpt_path = "/root/EAGLE/runs/colbert/best_model.ckpt"
    model2_ckpt_path = "/root/EAGLE/runs/eagle_weights/best_model.ckpt"

    # Read in the two result files
    result_file_1_path = os.path.join(
        eval_dir_path, model1, f"{tag1}_{dataset_name}_details.json"
    )
    result_file_2_path = os.path.join(
        eval_dir_path, model2, f"{tag2}_{dataset_name}_details.json"
    )
    logger.info(
        f"Reading result files from {result_file_1_path} and {result_file_2_path}"
    )
    results1 = file_utils.read_json_file(result_file_1_path)
    results2 = file_utils.read_json_file(result_file_2_path)
    results1 = {r[0]: r[1:] for r in results1}
    results2 = {r[0]: r[1:] for r in results2}
    assert len(results1) == len(
        results2
    ), f"Length of the two results are different {len(results1)} vs {len(results2)}"

    # Read in the query file
    logger.info(f"Reading query file from {query_path}")
    queries: List[Dict[str, Any]] = file_utils.read_jsonl_file(query_path)
    queries: Dict[str, str] = {q["_id"]: q["text"] for q in queries}
    # Reda in the corpus file
    logger.info(f"Reading corpus file from {corpus_path}")
    corpus: List[Dict[str, Any]] = file_utils.read_jsonl_file(corpus_path)
    corpus: Dict[str, Dict[str, str]] = {
        d["_id"]: {"text": d["text"], "title": d["title"]} for d in corpus
    }
    # Read in the dev file
    logger.info(f"Reading dev file from {dev_file_path}")
    dev_data = file_utils.read_jsonl_file(dev_file_path)

    # Compare
    same_qids: List[str] = []
    better1_qids: List[str] = []
    better2_qids: List[str] = []
    gold_pids_dic: Dict[str, List[int]] = {}
    for datum in tqdm.tqdm(dev_data):
        qid = datum["id"]
        gold_pids = datum["answers"]
        r1 = results1[qid]
        r2 = results2[qid]
        # Check the gold rank
        gold_pid = gold_pids[0]
        if gold_pid in r1[1]:
            gold_rank1 = r1[1].index(gold_pids[0])
        else:
            gold_rank1 = len(r1[1])
        if gold_pid in r2[1]:
            gold_rank2 = r2[1].index(gold_pids[0])
        else:
            gold_rank2 = len(r2[1])
        # Categorize the qid
        if gold_rank1 == gold_rank2:
            same_qids.append(qid)
        elif gold_rank1 < gold_rank2:
            better1_qids.append(qid)
        else:
            better2_qids.append(qid)
        # Aggregate gold pids
        gold_pids_dic[qid] = gold_pids

    # Load tokenizers
    logger.info(f"Creating tokenizers for {cfg.model.backbone_name}")
    q_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.query, model_name=cfg.model.backbone_name
    )
    d_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.document, model_name=cfg.model.backbone_name
    )
    # Load model 1
    logger.info(f"Loading model 1 from {model1_ckpt_path}")
    model1 = LightningNewModel.load_from_checkpoint(
        checkpoint_path=model1_ckpt_path, map_location="cuda:0"
    )
    model1.eval()

    # Load model 2
    logger.info(f"Loading model 2 from {model2_ckpt_path}")
    model2 = LightningNewModel.load_from_checkpoint(
        checkpoint_path=model2_ckpt_path, map_location="cuda:1"
    )
    model2.eval()

    # Examine what model1 did better than model2
    detailed_comparison_with_qids(
        model1=model1,
        model2=model2,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
        query_data=queries,
        corpus_data=corpus,
        qids_to_compare=better1_qids,
        results1=results1,
        results2=results2,
        gold_pids_dic=gold_pids_dic,
    )

    # Examine what model2 did better than model1
    stop = 1


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
