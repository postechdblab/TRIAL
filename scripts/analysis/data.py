from typing import *

import hkkang_utils.file as file_utils

DATASET_DIR = "/root/EAGLE/data"
DATASET_NAME = "msmarco"


def main():
    # Read in training queries
    train_file_path = "/root/EAGLE/data/msmarco_old/train_data_nhards256.jsonl"
    train_data: List[Dict] = file_utils.read_jsonl_file(train_file_path)

    queries_file_path = "/root/EAGLE/data/msmarco_old/queries.train.tsv"
    queries: List[Tuple[str, str]] = file_utils.read_csv_file(
        queries_file_path, delimiter="\t", first_row_as_header=False
    )
    queries: Dict[str, str] = {qid: query for qid, query in queries}

    # Read in collection
    collection = file_utils.read_csv_file(
        "/root/EAGLE/data/msmarco/collection.tsv",
        delimiter="\t",
        first_row_as_header=False,
        quotechar=None,
    )

    for i, datum in enumerate(train_data):
        qid = datum[0]
        doc_ids = datum[1:]
        print(f"\nQuery {i+1}: {queries[str(qid)]}")
        for j in range(10):
            if j == 0:
                doc_id = doc_ids[j] + 1
            else:
                doc_id = doc_ids[j + 50] + 1
            prefix = "Pos" if j == 0 else "Neg"
            print(f"Document {j+1} ({prefix}): {collection[doc_id]}")
            print()
        input()


if __name__ == "__main__":
    main()
