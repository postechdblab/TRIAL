import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import BaseRetriever, RetrievalResult

COLLECTION_PATH = "/root/ColBERT/data/msmarco_old/collection.tsv"
DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logger = logging.getLogger("MiniLLMRetriever")


class MiniLMRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_path: str = COLLECTION_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 256,
    ) -> None:
        # Set basic configs
        self.max_length = max_length

        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set the model
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # Load corpus
        super(MiniLMRetriever, self).__init__(corpus=self._load_corpus(corpus_path))

        # TODO: Need to encode and index the corpus
        pass

    def _load_corpus(
        self, corpus_path: str, use_cache: bool = True, overwrite_cache: bool = False
    ) -> List[str]:
        """Load the corpus from collection.tsv
        Each line of the collection.tsv is in the following format: doc_id \t doc_text
        """
        cache_path = corpus_path + ".cache"
        if use_cache and os.path.isfile(cache_path):
            logger.info(f"Loading corpus from {cache_path}")
            corpus: List[str] = file_utils.read_pickle_file(cache_path)
        else:
            logger.info(f"Loading corpus from {corpus_path}")
            corpus: List[List[str]] = file_utils.read_csv_file(
                corpus_path,
                delimiter="\t",
                first_row_as_header=False,
                show_progress=True,
            )
            corpus: List[str] = [item[1] for item in corpus]

        # Write the corpus to the cache
        if (use_cache and not os.path.isfile(cache_path)) or overwrite_cache:
            logger.info(f"Writing corpus of {len(corpus)} documents to {cache_path}")
            file_utils.write_pickle_file(corpus, cache_path)

        return corpus

    @torch.no_grad()
    def retrieve_batch(
        self, queries: List[str], topk: int = 100
    ) -> List[List[RetrievalResult]]:
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                # TODO: Need to implement retrieval
                raise NotImplementedError("Need to implement retrieval")

    @torch.no_grad()
    def calculate_score_by_text_batch(
        self, queries: List[str], doc_texts: List[str]
    ) -> List[float]:
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                # Tokenize queries and document texts
                features = self.tokenizer(
                    queries,
                    doc_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                ).to(self.model.device)
                # Get scores
                scores = self.model(**features).logits.squeeze(-1).tolist()
                return scores

    def create_index(self, *args, **kwargs) -> None:
        raise NotImplementedError("Cross encoder does not support create_index().")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    # Dummy query and text
    query = "What is the name of the dog?"
    text = "The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max. The dog's name is Max."

    # Initialize the retriever
    retriever = MiniLMRetriever()

    # Test the retriever
    # print(retriever.retrieve(query))
    logger.info(retriever.calculate_score(query=query, text=text))
    logger.info("Done!")
