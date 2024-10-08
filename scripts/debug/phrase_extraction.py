import tqdm
import hkkang_utils.file as file_utils
import logging
from eagle.phrase.constituency import ConstituencyParser

logger = logging.getLogger("DebugPhraseExtraction")


def examine() -> None:
    parser = ConstituencyParser()

    # Load data
    data_path = "/root/EAGLE/data/beir-msmarco/corpus.jsonl"
    logger.info(f"Loading dataset from {data_path} ...")
    dataset = file_utils.read_json_file(data_path, auto_detect_extension=True)
    logger.info(f"Dataset size: {len(dataset)}")
    results = []
    for datum in tqdm.tqdm(dataset[:100]):
        texts = datum["text"]
        parsed_sentences = parser(texts=texts, show_progress=True)
        results.append(parsed_sentences)
    return None


if __name__ == "__main__":
    examine()
