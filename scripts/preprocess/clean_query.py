import hkkang_utils.file as file_utils
import tqdm

from eagle.phrase.clean import unidecode_text

QUERY_PATH = "/root/EAGLE/data/beir-msmarco/queries.jsonl"
NEW_QUERY_PATH = "/root/EAGLE/data/beir-msmarco/queries.jsonl.clean"


def diff():
    old_queries = file_utils.read_jsonl_file(QUERY_PATH)
    new_queries = file_utils.read_jsonl_file(NEW_QUERY_PATH)

    for i, (old, new) in enumerate(zip(old_queries, new_queries)):
        if old["text"] != new["text"]:
            print(f"{i}: {old['text']} -> {new['text']}")


def main():
    queries = file_utils.read_jsonl_file(QUERY_PATH)

    # Clean
    for query in tqdm.tqdm(queries):
        query["text"] = unidecode_text(query["text"])

    # Write file
    file_utils.write_jsonl_file(queries, QUERY_PATH + ".clean")
    print("Done!")

    pass


if __name__ == "__main__":
    # main()
    diff()
