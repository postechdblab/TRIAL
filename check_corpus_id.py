import json

FILE_PATH = "/root/EAGLE/data/lotte-lifestyle-search/corpus.jsonl"


def main():
    # Read in the corpus.jsonl file
    with open(FILE_PATH, "r") as f:
        corpus = f.readlines()
    corpus = [json.loads(line) for line in corpus]

    # Check the id increment by 1
    for i in range(len(corpus)):
        if int(corpus[i]["_id"]) != i + 1:
            print(f"Error: id is not increment by 1 at index {i} ({corpus[i]['_id']})")
            break
    print("Check corpus id done")


if __name__ == "__main__":
    main()
