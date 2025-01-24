import hkkang_utils.file as file_utils
import tqdm


def main():
    data = file_utils.read_jsonl_file("/root/EAGLE/data/beir-trec-covid/dev.jsonl")
    answer_nums = []
    for datum in tqdm.tqdm(data):
        answer_num = len(datum["answers"])
        answer_nums.append(answer_num)
    print(answer_nums)
    # Average number of answers
    print(sum(answer_nums) / len(answer_nums))


if __name__ == "__main__":
    main()
