from typing import *
import os

import hkkang_utils.file as file_utils


def add_dummy_text(
    data: List[Dict[str, Any]], answer_pids: Set[int]
) -> List[Dict[str, Any]]:
    # Find the maximum idx
    max_idx = max([datum["_id"] for datum in data] + list(answer_pids))
    all_pids = set([datum["_id"] for datum in data])
    data_dic = {datum["_id"]: datum for datum in data}

    new_datum: List[Dict[str, Any]] = []
    for idx in range(1, max_idx + 1):
        if idx not in all_pids:
            print(f"Adding dummy text for idx: {idx}")
            new_datum.append({"_id": idx, "text": "", "title": ""})
        else:
            new_datum.append(data_dic[idx])
    return new_datum


def check_corpus(data: List[Dict[str, Any]], answer_pids: Set[int]) -> bool:
    for idx, datum in enumerate(data, start=1):
        datum_id = datum["_id"]
        if idx != datum_id:
            print(f"Missing idx: {idx}, datum_id: {datum_id}")
            if idx in answer_pids:
                print(f"idx {idx} is in answer_pids!!!")
            return False
    return True


def get_answer_pids(data: List[Dict[str, Any]]) -> Set[int]:
    answer_pids: List[int] = []
    for datum in data:
        answer_pids.extend(datum["answers"])
    return set(answer_pids)


def main():
    dataset_name = "beir-scidocs"
    root_path = "/root/EAGLE/data/"
    dataset_path = os.path.join(root_path, dataset_name, "corpus.jsonl")
    dev_path = os.path.join(root_path, dataset_name, "dev.jsonl")
    print(f"Reading dataset from {dataset_path}")
    dataset = file_utils.read_jsonl_file(dataset_path)
    print("Reading dev data from ")
    dev_data = file_utils.read_jsonl_file(dev_path)
    # Get the answer pids
    answer_pids = get_answer_pids(dev_data)
    # Check if the corpus is valid
    is_ok = check_corpus(dataset, answer_pids)

    # Add dummy text to fix the corpus
    if not is_ok:
        new_dataset = add_dummy_text(dataset, answer_pids)
        print(f"Writing new dataset to {dataset_path}")
        file_utils.write_jsonl_file(new_dataset, dataset_path)
    print("Done!")


if __name__ == "__main__":
    main()
