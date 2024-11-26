import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
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
    logger.info(f"{model1.model_name}: ")
    result1 = single_inference(
        model=model1,
        q_text=q_text,
        d_text=d_text,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
    )
    logger.info(f"{model2.model_name}: ")
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
        gold_pids: int = gold_pids_dic[qid_to_compare]
        gold_pid: int = gold_pids_dic[qid_to_compare][0]
        # Get the query and document texts
        query_sentences: List[str] = query_data[qid_to_compare]
        # Get the top-ranked document for model 1
        retrieved_pids1: List[int] = results1[qid_to_compare][1]
        retrieved_scores1: List[float] = results1[qid_to_compare][0]
        retrieved_pids2: List[int] = results2[qid_to_compare][1]
        retrieved_scores2: List[float] = results2[qid_to_compare][0]
        # Location of the gold pid in the retrieved pids
        model1_rank = (
            retrieved_pids1.index(gold_pid) + 1 if gold_pid in retrieved_pids1 else -1
        )
        model2_rank = (
            retrieved_pids2.index(gold_pid) + 1 if gold_pid in retrieved_pids2 else -1
        )
        doc_sentences: List[str] = corpus_data[gold_pid]["text"]
        doc_title: str = corpus_data[gold_pid]["title"]
        # Create query and doc text
        q_text = " ".join(query_sentences)
        gold_d_text = " ".join(doc_sentences)
        if doc_title:
            gold_d_text = doc_title + ". " + gold_d_text

        top_num_to_examine = 10
        retrieved_pids_to_examine1 = retrieved_pids1[:top_num_to_examine]
        retrieved_pids_to_examine2 = retrieved_pids2[:top_num_to_examine]
        retrieved_scores_to_examine1 = retrieved_scores1[:top_num_to_examine]
        retrieved_scores_to_examine2 = retrieved_scores2[:top_num_to_examine]
        pids_only_from_model1 = [
            item
            for item in retrieved_pids_to_examine1
            if item not in retrieved_pids_to_examine2
        ]
        pids_only_from_model2 = [
            item
            for item in retrieved_pids_to_examine2
            if item not in retrieved_pids_to_examine1
        ]

        # Print the information
        print("")
        logger.info(f"QID: {qid_to_compare}")
        logger.info(f"Gold pid: {gold_pid}")
        logger.info(f"Model1 Rank: {model1_rank}")
        logger.info(f"Model1 retrieved pids: {retrieved_pids_to_examine1}")
        logger.info(f"Model1 retrieved scores: {retrieved_scores_to_examine1}")
        logger.info(f"Model1 only pids: {pids_only_from_model1}")
        logger.info(f"Model2 Rank: {model2_rank}")
        logger.info(f"Model2 retrieved pids: {retrieved_pids_to_examine2}")
        logger.info(f"Model2 retrieved scores: {retrieved_scores_to_examine2}")
        logger.info(f"Model2 only pids: {pids_only_from_model2}")
        logger.info(f"Query: {q_text}")
        logger.info(f"Gold doc: {gold_d_text}")

        while True:
            pid_to_examine = input("Enter the pid to examine: ")
            if pid_to_examine == "exit":
                return None
            if pid_to_examine == "pass":
                break
            try:
                pid_to_examine = int(pid_to_examine)
            except:
                logger.error(f"Invalid pid: {pid_to_examine}")
                continue

            # Get the document text
            target_d_sentences: List[str] = corpus_data[pid_to_examine]["text"]
            target_d_title: str = corpus_data[pid_to_examine]["title"]
            target_d_text = " ".join(target_d_sentences)
            if target_d_title:
                target_d_text = doc_title + ". " + target_d_text

            detailed_comparison_with_qid(
                model1=model1,
                model2=model2,
                q_text=q_text,
                d_text=target_d_text,
                q_tokenizer=q_tokenizer,
                d_tokenizer=d_tokenizer,
            )
    return None


def load_model(model_ckpt_path: str) -> Tuple[LightningNewModel, DictConfig]:
    # Load model
    logger.info(f"Loading model from {model_ckpt_path}")
    model = LightningNewModel.load_from_checkpoint(
        checkpoint_path=model_ckpt_path, map_location="cuda:0"
    )
    model.eval()
    return model


@hydra.main(version_base=None, config_path="/root/EAGLE/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configs
    dataset_name = "beir-arguana"
    model1_ckpt_path = "/root/EAGLE/runs/colbert/best_model.ckpt"
    model2_ckpt_path = "/root/EAGLE/runs/eagle_weights/best_model.ckpt"
    cfg.tokenizers.query.max_len = 300
    # model1_ckpt_path = cfg.args.ckpt1
    # model2_ckpt_path = cfg.args.ckpt2
    # dataset_name = cfg.dataset.name

    eval_dir_path = cfg._global.eval_dir
    query_path: str = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.query_file
    )
    corpus_path: str = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.corpus_file
    )
    dev_file_path = os.path.join(
        cfg.dataset.dir_path, dataset_name, cfg.dataset.val.data_file
    )

    # Load tokenizers
    logger.info(f"Creating tokenizers for {cfg.model.backbone_name}")
    q_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.query, model_name=cfg.model.backbone_name
    )
    d_tokenizer = Tokenizer(
        cfg=cfg.tokenizers.document, model_name=cfg.model.backbone_name
    )
    # Load model 1
    model1 = load_model(model1_ckpt_path)
    model2 = load_model(model2_ckpt_path)

    model1_name = model1.model.cfg.name
    model2_name = model2.model.cfg.name

    tag1 = model1.cfg.tag
    tag2 = model2.cfg.tag

    # Read in the two result files
    result_file_1_path = os.path.join(
        eval_dir_path, model1_name, f"{tag1}_{dataset_name}_details.json"
    )
    result_file_2_path = os.path.join(
        eval_dir_path, model2_name, f"{tag2}_{dataset_name}_details.json"
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

    # Examine what model1 did better than model2
    detailed_comparison_with_qids(
        model1=model1,
        model2=model2,
        q_tokenizer=q_tokenizer,
        d_tokenizer=d_tokenizer,
        query_data=queries,
        corpus_data=corpus,
        qids_to_compare=better2_qids,
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
