from typing import List

DATASET_NAMES = [
    # "beir-arguana",
    # "beir-climate-fever",
    # "beir-dbpedia-entity",
    # "beir-fever",
    # "beir-fiqa",
    # "beir-hotpotqa",
    # "beir-msmarco",
    # "beir-nfcorpus",
    # "beir-nq",
    # "beir-quora",
    # "beir-scidocs",
    # "beir-scifact",
    # "beir-trec-covid",
    # "beir-webis-touche2020",
    "lotte-lifestyle-forum",
    "lotte-lifestyle-search",
    "lotte-pooled-forum",
    "lotte-pooled-search",
    "lotte-recreation-forum",
    "lotte-recreation-search",
    "lotte-science-forum",
    "lotte-science-search",
    "lotte-technology-forum",
    "lotte-technology-search",
    "lotte-writing-forum",
    "lotte-writing-search",
]


def safe_divide(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b


def avg(lst: List[int]) -> float:
    return safe_divide(sum(lst), len(lst))


def avg_list_of_list(lst: List[List[int]], agg_func=len) -> float:
    return safe_divide(sum([agg_func(item) for item in lst]), len(lst))
