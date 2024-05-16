import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.slack as slack_utils
import tqdm

from model.late_encoder import ColBERTRetriever
from model.utils import Document
from scripts.evaluate.utils import load_beir_data
from scripts.utils import BEIR_DATASET_NAMES, validate_model_name

# Retriever path
ROOT = "/root/ColBERT/experiments/"
DATASET_DIR = "/root/ColBERT/data"
CHECKPOINT_DIR = "/root/ColBERT/checkpoint"
NBITS = 2

logger = logging.getLogger("Evaluate")


def load_data(
    dataset_name: str, filter_type: str = None, sample_num: int = None
) -> List:
    """
    List[Tuple[str, str, List[str], List[str]]
    - Tuple: (qid, query, pids, p_titles)
    """

    data = load_beir_data(
        dataset_dir=DATASET_DIR, dataset_name=dataset_name, return_unique=True
    )

    # Sample data
    if sample_num:
        data = data[:sample_num]
        logger.info(f"Filtered data: {len(data)}")

    return data


def main(
    dataset_name: str,
    model_name: str,
    metric: str,
    skip_padding: bool = False,
    use_phrase_level: bool = False,
    save_result: bool = False,
    save_wrong: bool = False,
    save_dir: str = None,
    filter_type: str = None,
    sample_num: int = None,
    is_unidecode: bool = True,
    max_q_len: int = None,
    tag: Optional[str] = None,
):
    """Evaluate the retrieval performance of ColBERT."""
    # Load dataset
    data: List = load_data(
        dataset_name=dataset_name, filter_type=filter_type, sample_num=sample_num
    )

    # Initialize retriever
    logger.info(f"Initializing retriever...")
    index = f"{dataset_name}.{model_name}.nbits={NBITS}"
    experiment = f"{dataset_name}_unidecode" if is_unidecode else dataset_name
    retriever = ColBERTRetriever(
        root=ROOT,
        index=index,
        experiment=experiment,
        use_cache=False,
        skip_padding=skip_padding,
        is_use_phrase_level=use_phrase_level,
    )
    if max_q_len:
        retriever.searcher.config.query_maxlen = max_q_len

    # Get queries without duplicates
    indices = []
    queries = []
    qid_set = set()
    for i, (qid, query, pids, p_titles, p_scores) in enumerate(tqdm.tqdm(data)):
        if qid not in qid_set:
            indices.append(i)
            qid_set.add(qid)
            queries.append(query)

    # Search for top-k passages for each query
    logger.info(f"Searching top-k passages for {len(queries)} query...")
    initial_pids, final_pids = retriever.retrieve_candidates_batch(
        queries=queries, topk=100
    )
    assert len(initial_pids) == len(
        indices
    ), f"Different number of queries and initial pids: {len(initial_pids)} vs {len(indices)}"
    assert len(final_pids) == len(
        indices
    ), f"Different number of queries and final pids: {len(final_pids)} vs {len(indices)}"

    # Create Dictionary for the candidate documents
    final_candidate_dict = {}
    initial_candidate_dict = {}
    for i, idx in enumerate(indices):
        qid, query, pids, p_titles, p_scores = data[idx]
        initial_candidate_dict[qid] = initial_pids[i]
        final_candidate_dict[qid] = final_pids[i]

    # Configure metric
    if metric == "all":
        is_correct_func = all
    elif metric == "any":
        is_correct_func = any
    else:
        raise ValueError(f"Invalid metric: {metric}")

    # Evaluate the retrieval performance
    logger.info(f"Evaluating...")
    is_not_in_initial_candidates: List[str] = []
    is_not_in_final_candidates: List[str] = []
    for qid, query, pids, p_titles, p_scores in data:
        positive_pids = [str(pid) for pid in pids]
        # Get top-k negative passages
        initial_candidates: List[Document] = initial_candidate_dict[qid]
        initial_candidates: List[str] = [doc.id for doc in initial_candidates]
        final_candidates: List[Document] = final_candidate_dict[qid]
        final_candidates: List[str] = [doc.id for doc in final_candidates]

        # Check if the all positive docs are included in the initial candidates
        assert positive_pids, f"No positive passages for query {qid}"
        if not is_correct_func([pid in initial_candidates for pid in positive_pids]):
            is_not_in_initial_candidates.append(qid)
        if not is_correct_func([pid in final_candidates for pid in positive_pids]):
            is_not_in_final_candidates.append(qid)

    # Print the results
    logger.info(f"Total number of queries: {len(data)}")
    logger.info(
        f"Not included in the initial candidates: {len(is_not_in_initial_candidates)}"
    )
    logger.info(
        f"Not included in the final candidates: {len(is_not_in_final_candidates)}"
    )

    # Write wrong list
    if save_wrong:
        logger.info(f"Saving wrong list...")
        # Create the output file name
        output_file_name = f"candidate_wrong_list.{dataset_name}_{model_name}_{metric}"
        if tag:
            output_file_name += f"_{tag}"
        output_file_name += ".json"
        # Write to file
        file_utils.write_json_file(
            list(is_not_in_initial_candidates),
            os.path.join(save_dir, f"init_{output_file_name}"),
        )
        file_utils.write_json_file(
            list(is_not_in_final_candidates),
            os.path.join(save_dir, f"final_{output_file_name}"),
        )

    # Write candidate results
    if save_result:
        logger.info(f"Saving results...")
        # Create the output file name
        output_file_name = f"candidate_results.{dataset_name}_{model_name}_{metric}"
        if tag:
            output_file_name += f"_{tag}"
        output_file_name += ".json"
        # Convert to JSON
        stringified_initial_candidates: Dict[str, str] = {}
        stringified_final_candidates: Dict[str, str] = {}
        for qid in initial_candidate_dict.keys():
            initial_pids: List[Document] = initial_candidate_dict[qid]
            final_pids: List[Document] = final_candidate_dict[qid]
            stringified_initial_candidates[qid] = [pid.dic for pid in initial_pids]
            stringified_final_candidates[qid] = [pid.dic for pid in final_pids]
        # Write to file
        file_utils.write_json_file(
            stringified_initial_candidates,
            os.path.join(save_dir, f"init_{output_file_name}"),
        )
        file_utils.write_json_file(
            stringified_final_candidates,
            os.path.join(save_dir, f"final_{output_file_name}"),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Start end-to-end test.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate",
        choices=BEIR_DATASET_NAMES + ["all"],
        default="fever",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model checkpoint to evaluate",
        # default="baseline_nway64_q4"
        default="colbertv2.0",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to use for evaluation (All: correct if all gold included, Any: correct if any gold included)",
        choices=["all", "any"],
        default="all",
    )
    parser.add_argument(
        "--is_phrase",
        action="store_true",
        help="Whether to use phrase level",
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
        "--max_q_len", type=int, help="Maximum length of query", default=None
    )
    parser.add_argument(
        "--slack", action="store_true", help="Whether to send slack notification"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if arguments are valid
    validate_model_name(CHECKPOINT_DIR, args.model)

    # Create output directory if not exists
    if (args.save_result or args.save_wrong) and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.save_dir:
        file_handler_path = (
            f"token_retrieval_{args.dataset}_{args.model}_{args.metric}.log"
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
        success_msg=f"Succeeded to evaluate token retrieval for {args.model} on {args.dataset}",
        error_msg=f"Falied to evaluate token retrieval for {args.model} on {args.dataset}",
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
                metric=args.metric,
                skip_padding=args.skip_padding,
                use_phrase_level=args.is_phrase,
                is_unidecode=not args.no_unidecode,
                save_result=args.save_result,
                save_wrong=args.save_wrong,
                save_dir=args.save_dir,
                filter_type=args.filter_type,
                sample_num=args.sample_num,
                max_q_len=args.max_q_len,
                tag=args.save_tag,
            )

    logger.info(f"Done!")
