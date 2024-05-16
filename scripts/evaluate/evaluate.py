import argparse
import copy
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.slack as slack_utils
import tqdm
from beir.retrieval.custom_metrics import mrr
from beir.retrieval.evaluation import EvaluateRetrieval

from model import RetrievalResult
from model.late_encoder import ColBERTRetriever
from scripts.evaluate.utils import (
    data_to_beir_format,
    load_data,
    results_to_beir_format,
)
from scripts.utils import BEIR_DATASET_NAMES, validate_model_name

# Retriever path
ROOT = "/root/ColBERT/experiments/"
DATASET_DIR = "/root/ColBERT/data"
CHECKPOINT_DIR = "/root/ColBERT/checkpoint"
NBITS = 2

logger = logging.getLogger("Evaluate")


def main(
    dataset_name: str,
    model_name: str,
    skip_padding: bool = False,
    use_phrase_level: bool = False,
    use_min_threshold: bool = False,
    use_noun_importance: bool = False,
    noun_importance_weight: float = 1.5,
    save_result: bool = False,
    save_wrong: bool = False,
    save_score: bool = False,
    save_dir: str = None,
    filter_type: str = None,
    sample_num: int = None,
    is_unidecode: bool = True,
    is_oracle_candidate: bool = False,
    is_include_gold: bool = False,
    is_full_length_search: bool = False,
    max_q_length: Optional[int] = None,
    max_d_length: Optional[int] = None,
    tag: Optional[str] = None,
):
    """Evaluate the retrieval performance of ColBERT."""
    # Load dataset
    data: List = load_data(
        dataset_dir=DATASET_DIR,
        dataset_name=dataset_name,
        filter_type=filter_type,
        sample_num=sample_num,
    )

    # Initialize retriever
    logger.info(f"Initializing retriever...")
    index = f"{dataset_name}.{model_name}.nbits={NBITS}"
    experiment = f"{dataset_name}_unidecode" if is_unidecode else dataset_name
    retriever = ColBERTRetriever(
        root=ROOT,
        index=index,
        experiment=experiment,
        use_cache=True,
        skip_padding=skip_padding,
        is_use_phrase_level=use_phrase_level,
        is_use_min_threshold=use_min_threshold,
        is_noun_important=use_noun_importance,
        noun_importance_weight=noun_importance_weight,
    )

    # Override configs
    if max_q_length is not None:
        retriever.config.query_maxlen = max_q_length
        retriever.searcher.config.query_maxlen = max_q_length
        retriever.searcher.checkpoint.query_tokenizer.query_maxlen = max_q_length
    if max_d_length is not None:
        retriever.config.doc_maxlen = max_d_length
        retriever.searcher.config.doc_maxlen = max_d_length
        retriever.searcher.checkpoint.doc_tokenizer.doc_maxlen = max_d_length

    # Get queries without duplicates
    indices = []
    queries = []
    qid_set = set()
    gold_pids_dict = dict()
    for i, (qid, query, pids, p_titles, scores) in enumerate(tqdm.tqdm(data)):
        if qid not in qid_set:
            indices.append(i)
            qid_set.add(qid)
            queries.append(query)
        if qid not in gold_pids_dict:
            gold_pids_dict[qid] = copy.deepcopy(pids)
        else:
            gold_pids_dict[qid].extend(pids)

    # Remove redundant pids and convert to int
    for qid in gold_pids_dict:
        gold_pids_dict[qid] = [int(item) for item in set(gold_pids_dict[qid])]

    # Get the gold pids for each query
    required_candidates = None
    if is_oracle_candidate:
        required_candidates: List[List] = []
        for item_idx in indices:
            qid = data[item_idx][0]
            required_candidates.append(gold_pids_dict[qid])

    required_pids = None
    if is_include_gold:
        required_pids: List[List] = []
        for item_idx in indices:
            qid = data[item_idx][0]
            required_pids.append(gold_pids_dict[qid])

    # Search for top-k passages for each query
    logger.info(f"Searching top-k passages for {len(queries)} query...")
    topk_passages_list: List[List[RetrievalResult]] = retriever.retrieve_batch(
        queries=queries,
        topk=1000,
        return_scores=save_score,
        required_pids=required_pids,
        required_candidates=required_candidates,
        is_full_length_search=is_full_length_search,
    )
    assert len(topk_passages_list) == len(
        indices
    ), f"Different number of queries and topk passages: {len(topk_passages_list)} vs {len(indices)}"
    topk_passages_dict: Dict[str, List[RetrievalResult]] = {
        index: topk_passages
        for index, topk_passages in zip(indices, topk_passages_list)
    }

    # Get document ids
    new_topk_passages_dict = {}
    for key, results in topk_passages_dict.items():
        qid = data[key][0]
        assert qid not in new_topk_passages_dict, f"Duplicate qid: {qid}"
        new_topk_passages_dict[qid] = results
    topk_passages_dict = new_topk_passages_dict

    # Evaluate the retrieval performance
    logger.info(f"Evaluating...")
    top_5_recall = []
    top_10_recall = []
    top_50_recall = []
    top_100_recall = []
    top_1000_recall = []
    wrong_list = set()
    for qid, query, pids, p_titles, p_scores in data:
        positive_pids = [str(pid) for pid in pids]
        # Get top-k negative passages
        topk_passages = topk_passages_dict[qid]
        topk_passages = [str(result.doc.id) for result in topk_passages]
        # Evaluate recall @ 10, @50, @100
        is_correct = all([pid in topk_passages[:50] for pid in positive_pids])
        if not is_correct:
            wrong_list.add(qid)
        top_5_recall.append(all([pid in topk_passages[:5] for pid in positive_pids]))
        top_10_recall.append(all([pid in topk_passages[:10] for pid in positive_pids]))
        top_50_recall.append(all([pid in topk_passages[:50] for pid in positive_pids]))
        top_100_recall.append(
            all([pid in topk_passages[:100] for pid in positive_pids])
        )
        top_1000_recall.append(
            all([pid in topk_passages[:1000] for pid in positive_pids])
        )

    # Print the results
    logger.info(f"Total number of queries: {len(data)}")
    logger.info(f"Recall@5: {sum(top_5_recall)/len(top_5_recall)}")
    logger.info(f"Recall@10: {sum(top_10_recall)/len(top_10_recall)}")
    logger.info(f"Recall@50: {sum(top_50_recall)/len(top_50_recall)}")
    logger.info(f"Recall@100: {sum(top_100_recall)/len(top_100_recall)}")
    logger.info(f"Recall@1000: {sum(top_1000_recall)/len(top_1000_recall)}")

    # Evaluate using BEIR script
    qrels = data_to_beir_format(data)

    # Arguana corpus contains the exact same passage with the query
    if dataset_name == "arguana":
        topk_passages_list = [item[1:] for item in topk_passages_list]

    results = results_to_beir_format(
        qids=[data[idx][0] for idx in indices], results=topk_passages_list
    )
    k_values = [1, 3, 5, 10, 50, 100, 1000]
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    _mrr = mrr(qrels, results, k_values)

    # Write wrong list
    if save_wrong:
        logger.info(f"Saving {len(wrong_list)} wrong list...")
        output_file_name = f"wrong_list.{dataset_name}_{model_name}.json"
        if tag:
            output_file_name = f"wrong_list.{dataset_name}_{model_name}_{tag}.json"
        file_utils.write_json_file(
            list(wrong_list), os.path.join(save_dir, output_file_name)
        )
    if save_result:
        logger.info(f"Saving {len(topk_passages_list)} results...")
        # Convert to JSON
        result_dict: Dict[str, List[RetrievalResult]] = {}
        for qid, query, pids, p_titles, p_scores in data:
            topk_passages = topk_passages_dict[qid]
            # Get the results for the gold passages
            gold_passages = [item for item in topk_passages if str(item.doc.id) in pids]
            assert (
                len(gold_passages) > 0
            ), f"No gold passages found in the retrieved set (qid: {qid}, pids: {pids})"
            result_dict[qid] = (pids, gold_passages, topk_passages[:50])
        output_file_name = f"result.{dataset_name}_{model_name}.pkl"
        if tag:
            output_file_name = f"result.{dataset_name}_{model_name}_{tag}.pkl"
        file_utils.write_pickle_file(
            result_dict, os.path.join(save_dir, output_file_name)
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate",
        choices=BEIR_DATASET_NAMES + ["all"],
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
        "--save_wrong", action="store_true", help="Whether to save the wrong list"
    )
    parser.add_argument(
        "--save_result", action="store_true", help="Whether to save the result"
    )
    parser.add_argument(
        "--save_score", action="store_true", help="Whether to save the scores"
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
        "--oracle_candidate",
        action="store_true",
        help="Whether to include gold during reranking",
    )
    parser.add_argument(
        "--include_gold",
        action="store_true",
        help="Whether to ensure gold is in the top-k retrieval result",
    )
    parser.add_argument(
        "--full_length_search",
        action="store_true",
        help="Whether to search the full length",
    )
    parser.add_argument(
        "--max_q_length", type=int, help="Maximum query length", default=None
    )
    parser.add_argument(
        "--max_d_length", type=int, help="Maximum document length", default=None
    )
    parser.add_argument(
        "--slack", action="store_true", help="Whether to send slack notification"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save_score:
        assert args.save_result, "save_result must be True to save scores"
    if args.save_result:
        assert args.include_gold, "include_gold must be True to save results"

    # Check if arguments are valid
    validate_model_name(CHECKPOINT_DIR, args.model)

    # Create output directory if not exists
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.save_dir:
        file_handler_path = (
            f"eval_{args.dataset}_{args.model}_{args.oracle_candidate}.log"
        )
        if args.save_tag:
            file_handler_path = file_handler_path.replace(
                ".log", f"_{args.save_tag}.log"
            )
        file_handler_path = os.path.join(args.save_dir, file_handler_path)

    # Set logging
    handlers = [logging.StreamHandler()]
    if args.save_dir:
        handlers.append(logging.FileHandler(file_handler_path))
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )

    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to evaluate {args.model} on {args.dataset} dataset",
        error_msg=f"Falied to evaluate {args.model} on {args.dataset} dataset",
        disable=not args.slack,
    ):
        # Configure evaluation dataset
        if args.dataset == "all":
            eval_datasets = BEIR_DATASET_NAMES
        else:
            eval_datasets = [args.dataset]

        # Evaluate each dataset
        for dataset_name in tqdm.tqdm(eval_datasets, desc="Evaluating datasets"):
            logger.info(f"Evaluating {dataset_name}...")
            main(
                dataset_name=dataset_name,
                model_name=args.model,
                skip_padding=args.skip_padding,
                use_phrase_level=args.is_phrase,
                use_min_threshold=args.is_min_threshold,
                use_noun_importance=args.is_noun_important,
                noun_importance_weight=args.noun_importance_weight,
                is_unidecode=not args.no_unidecode,
                save_result=args.save_result,
                save_wrong=args.save_wrong,
                save_score=args.save_score,
                save_dir=args.save_dir,
                filter_type=args.filter_type,
                sample_num=args.sample_num,
                is_oracle_candidate=args.oracle_candidate,
                is_include_gold=args.include_gold,
                is_full_length_search=args.full_length_search,
                max_q_length=args.max_q_length,
                max_d_length=args.max_d_length,
                tag=args.save_tag,
            )

    logger.info(f"Done!")
