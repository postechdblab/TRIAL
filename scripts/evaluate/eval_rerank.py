import copy
import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.slack as slack_utils
import tqdm

from model.late_encoder import ColBERTRetriever
from scripts.evaluate.utils import get_recall_rates
from scripts.utils import read_queries, validate_model_name

logger = logging.getLogger("EvalReranker")

# Retriever path
ROOT = "/root/ColBERT/experiments/"
DATASET_DIR = "/root/ColBERT/data"
CHECKPOINT_DIR = "/root/ColBERT/checkpoint"
NBITS = 2

logger = logging.getLogger("EvaluateRerank")


def load_data_hotpotqa() -> List:
    dataset_path = f"{DATASET_DIR}/hotpotqa_old/dev_data.jsonl"
    query_path = f"{DATASET_DIR}/hotpotqa_old/dev.json"
    query_dict = file_utils.read_json_file(query_path)
    query_dict = {str(item["id"]): item["question"] for item in query_dict}
    data: List = file_utils.read_jsonl_file(dataset_path)

    parsed_data = []
    for i in tqdm.tqdm(range(len(data))):
        datum = data[i]
        qid = datum[0]
        gold_pids = list(set(datum[1:4]))
        neg_pids = datum[4:]
        # Get the query string
        query = query_dict[str(qid)]
        parsed_data.append((qid, query, gold_pids, neg_pids))
    return parsed_data


def load_data_msmarco() -> List:
    dataset_path = f"{DATASET_DIR}/msmarco_old/dev_data.jsonl"
    query_path = f"{DATASET_DIR}/msmarco_old/queries.dev.tsv"
    query_dict = read_queries(query_path)
    data: List = file_utils.read_jsonl_file(dataset_path)

    # Rank documents
    parsed_data = []
    for i in tqdm.tqdm(range(len(data))):
        datum = data[i]
        qid = datum[0]
        gold_pids = [datum[1]]
        neg_pids = datum[2:]
        # Get the query string
        query = query_dict[str(qid)]
        parsed_data.append((qid, query, gold_pids, neg_pids))

    return parsed_data


def load_data(
    dataset_name: str,
    filter_type: Optional[str] = None,
    sample_num: Optional[int] = None,
) -> List:
    if dataset_name == "msmarco":
        data = load_data_msmarco()
    elif dataset_name == "hotpotqa":
        data = load_data_hotpotqa()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    # Sample data
    if sample_num:
        data = data[:sample_num]
        logger.info(f"Sampled data: {len(data)}")

    if filter_type:
        # Filter data that contains the filter_type in the query text
        logger.info(f"Filtering {len(data)} data with {filter_type}...")
        data = [item for item in data if filter_type in item[1]]
        logger.info(f"Filtered data: {len(data)}")

    return data


def main(
    dataset_name: str,
    model_name: str,
    skip_padding: bool = False,
    use_phrase_level: bool = False,
    use_min_threshold: bool = False,
    use_weighted_sum: bool = False,
    use_noun_importance: bool = False,
    noun_importance_weight: float = 1.5,
    save_result: bool = False,
    save_wrong: bool = False,
    save_dir: str = None,
    filter_type: str = None,
    sample_num: int = None,
    is_unidecode: bool = True,
    tag: Optional[str] = None,
) -> None:
    # Load dataset
    logger.info(f"Reading data...")
    data: List = load_data(
        dataset_name=dataset_name, filter_type=filter_type, sample_num=sample_num
    )

    # Load model
    logger.info(f"Initializing retriever...")
    # index = f"{dataset_name}.{model_name}.nbits={NBITS}"
    checkpoint = os.path.join(CHECKPOINT_DIR, model_name)
    experiment = f"{dataset_name}_unidecode" if is_unidecode else dataset_name
    corpus_path = f"{DATASET_DIR}/{dataset_name}_old/collection.tsv"
    retriever = ColBERTRetriever(
        root=ROOT,
        index=None,
        experiment=experiment,
        checkpoint_path=checkpoint,
        corpus_path=corpus_path,
        skip_loading=True,
        use_cache=True,
        skip_padding=skip_padding,
        is_use_phrase_level=use_phrase_level,
        is_use_min_threshold=use_min_threshold,
        is_use_weighted_sum=use_weighted_sum,
        is_noun_important=use_noun_importance,
        noun_importance_weight=noun_importance_weight,
    )

    # Rank documents
    all_rank_results: List[List[Tuple[int, int]]] = []
    for i in tqdm.tqdm(range(len(data))):
        datum = data[i]
        query = datum[1]
        gold_pids = datum[2]
        neg_pids = datum[3]
        # Combine without duplicate
        pids = copy.deepcopy(gold_pids)
        for pid in neg_pids:
            if pid not in pids:
                pids.append(pid)
        # Get the query string
        rank_results: List[Tuple[int, int]] = retriever.rank_docs(
            query=query, pids=pids
        )
        all_rank_results.append(rank_results)

    # Evaluate the ranking results
    dicts = {}
    for i in range(len(data)):
        rank_results = all_rank_results[i]
        gold_pids = data[i][2]
        ranked_pids = [pid for score, pid in rank_results]
        recall = get_recall_rates(ranked_pids=ranked_pids, gold_pids=gold_pids)
        for key, value in recall.items():
            if key not in dicts:
                dicts[key] = []
            dicts[key].append(value)
    # Average the recall rates
    for key, value in dicts.items():
        logger.info(f"{key}: {sum(value)/len(value)}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate",
        choices=["msmarco", "hotpotqa"],
        default="trec-covid",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model checkpoint to evaluate",
        # default="baseline_nway64_q4"
        default="colbertv2.0",
    )
    parser.add_argument(
        "--is_phrase",
        action="store_true",
        help="Whether to use phrase level",
    )
    parser.add_argument(
        "--is_min_threshold",
        action="store_true",
        help="Whether to use minimum threshold",
    )
    parser.add_argument(
        "--is_weighted_sum",
        action="store_true",
        help="Whether to use weighted sum",
    )
    parser.add_argument(
        "--is_noun_important",
        action="store_true",
        help="Whether to use noun importance",
    )
    parser.add_argument(
        "--noun_importance_weight",
        type=float,
        help="Weight for noun importance",
        default=1.5,
    )
    parser.add_argument(
        "--save_wrong", action="store_true", help="Whether to return the wrong list"
    )
    parser.add_argument(
        "--save_result", action="store_true", help="Whether to return the wrong list"
    )
    parser.add_argument("--save_tag", type=str, help="Tag of output file", default="")
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save the result", default=None
    )
    parser.add_argument(
        "--filter_type", type=str, help="Type of data to filter", default=None
    )
    parser.add_argument(
        "--sample_num", type=int, help="Number of samples", default=None
    )
    parser.add_argument(
        "--skip_padding", action="store_true", help="Whether to skip padding"
    )
    parser.add_argument(
        "--no_unidecode", action="store_true", help="Whether to use unidecode"
    )
    parser.add_argument(
        "--slack", action="store_true", help="Whether to send slack notification"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    # Check if arguments are valid
    validate_model_name(CHECKPOINT_DIR, args.model)

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to evaluate reranking {args.model} on {args.dataset} dataset",
        error_msg=f"Falied to evaluate reranking {args.model} on {args.dataset} dataset",
        disable=not args.slack,
    ):
        main(
            dataset_name=args.dataset,
            model_name=args.model,
            skip_padding=args.skip_padding,
            use_phrase_level=args.is_phrase,
            use_min_threshold=args.is_min_threshold,
            use_weighted_sum=args.is_weighted_sum,
            use_noun_importance=args.is_noun_important,
            noun_importance_weight=args.noun_importance_weight,
            save_result=args.save_result,
            save_wrong=args.save_wrong,
            save_dir=args.save_dir,
            filter_type=args.filter_type,
            sample_num=args.sample_num,
            is_unidecode=not args.no_unidecode,
            tag=args.save_tag,
        )

    logger.info(f"Done!")
