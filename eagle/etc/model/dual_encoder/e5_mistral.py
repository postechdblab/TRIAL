import functools
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.slack as slack_utils
import torch
import tqdm
import ujson
from transformers import AutoModel, AutoTokenizer

from model import BaseRetriever, RetrievalResult
from model.dual_encoder.utils import (
    get_detailed_instruct,
    last_token_pool,
    task_name_to_instruct,
)

COLLECTION_PATH = "/root/EAGLE/data/beir-msmarco/corpus.jsonl"
DEFAULT_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"

logger = logging.getLogger("E5MistralRetriever")


class E5MistralRetriever(BaseRetriever):
    def __init__(
        self,
        corpus_path: str = COLLECTION_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 4096,
        dataset_name: str = "MSMARCO",
        skip_loading_model: bool = False,
    ) -> None:
        """For details, please refer to https://huggingface.co/intfloat/e5-mistral-7b-instruct"""
        assert (
            dataset_name in task_name_to_instruct
        ), f"dataset_name ({dataset_name}) is not supported"
        # Set basic configs
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.corpus_path = corpus_path

        self.model = (
            None
            if skip_loading_model
            else AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            )
        )
        # self.model = None if skip_loading_model else AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set the model
        if self.model:
            # torch.set_float32_matmul_precision('high')
            # logger.info(f"Compiling the model {model_name}")
            # self.model = torch.compile(self.model, mode="max-autotune")
            # logger.info(f"Compile done!")
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()

        # Load corpus
        super(E5MistralRetriever, self).__init__(corpus=self._load_corpus(corpus_path))
        pass

    @property
    def instruction(self) -> str:
        return task_name_to_instruct[self.dataset_name]

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
            corpus: List[List[str]] = file_utils.read_jsonl_file(
                corpus_path,
            )
            corpus: List[str] = [item["text"] for item in corpus]

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
                pass

    @torch.no_grad()
    def calculate_score_by_text_batch(
        self, queries: List[str], doc_texts: List[str]
    ) -> List[float]:
        """Calculate the similarity scores between the given queries and the given documents."""
        with torch.inference_mode():
            # Prepare the input texts
            get_detailed_instructions = functools.partial(
                get_detailed_instruct, self.instruction
            )
            instructions = list(map(get_detailed_instructions, queries))

            # Get the representation vectors for the queries
            query_embeddings = self.get_representation_vector_batch(instructions)
            # Get the representation vectors for the documents
            doc_embeddings = self.get_representation_vector_batch(doc_texts)

            assert len(query_embeddings) == len(
                doc_embeddings
            ), f"Number of queries ({len(query_embeddings)}) and documents ({len(doc_embeddings)}) are not the same"

            # Batch-wise compute dot-product similarity (actually cosine similarity becauese the embeddings are normalized)
            scores = (
                torch.bmm(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(2))
                .squeeze(-1)
                .squeeze(-1)
                * 100
            )
            # Format the output scores
            return scores.tolist()

    def get_representation_vector_batch(self, input_texts: List[str]) -> torch.Tensor:
        """Get the representation vectors for the given input texts.
        The representation vectors are the embeddings of the last token of the input texts and they are normalized.
        """
        # Tokenize
        batch_dict = self.tokenizer(
            input_texts,
            max_length=self.max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # Append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        # Pad
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        # Forward
        outputs = self.model(
            batch_dict["input_ids"].to(self.model.device),
            attention_mask=batch_dict["attention_mask"].to(self.model.device),
        )
        # Get embeddings
        embeddings = last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    @torch.no_grad()
    def create_index(self, index_name: str, corpus_path: str) -> None:
        """Create the index for the corpus."""
        # Configs
        # offset=0
        # max_offset = 4000000
        offset = 4000000
        max_offset = 0
        cluster_size = 100000

        # Create directory for the index
        index_dir = os.path.join("index", self.__class__.__name__, index_name)
        os.makedirs(index_dir, exist_ok=True)

        # Read the corpus
        assert (
            self.corpus_path == corpus_path
        ), f"corpus_path ({corpus_path}) is not the same as the corpus_path used in __init__ ({self.corpus_path})"
        corpus = self.corpus
        # Set the max_offset
        if max_offset == 0:
            max_offset = len(corpus)

        for c_idx in tqdm.tqdm(
            range(offset, min(len(corpus), max_offset), cluster_size),
            desc="Creating index",
        ):
            # Get the cluster
            logger.info(
                f"Creating the index for the cluster {c_idx} to {min(c_idx + cluster_size, len(corpus))}"
            )
            subcorpus = corpus[c_idx : c_idx + cluster_size]

            # Check if cache file exists
            cache_file_path = os.path.join(index_dir, f"c_{c_idx}.cache")
            # Read the cache if exists
            if os.path.isfile(cache_file_path):
                logger.info(f"Loading the index from {cache_file_path} ...")
                index: Dict[str, List[float]] = {}
                with open(cache_file_path, "r") as f:
                    for line in tqdm.tqdm(f):
                        idx, embedding = ujson.loads(line)
                        index[idx] = embedding
                logger.info(f"Loaded {len(index)} documents from {cache_file_path}")
            else:
                index = {}

            # Remove the documents that are already indexed
            logger.info(f"Removing the documents that are already indexed")
            target_subcorpus: List[str] = []
            for idx, doc in enumerate(subcorpus):
                if str(idx) not in index:
                    target_subcorpus.append(doc)
            logger.info(
                f"Number of documents to be indexed:{len(subcorpus)} -> {len(target_subcorpus)}"
            )

            # Tokenize the corpus
            logger.info(f"Tokenizing {len(target_subcorpus)} documents")
            tokenized_corpus: List[int] = self.tokenizer(
                target_subcorpus,
                max_length=self.max_length - 1,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )["input_ids"]
            tokenized_corpus = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in tokenized_corpus
            ]

            # Sort the tokenized corpus by the length of each document (save the idx)
            tokenized_corpus: List[Tuple[int, List[int]]] = [
                (idx, doc) for idx, doc in enumerate(tokenized_corpus)
            ]
            tokenized_corpus.sort(key=lambda x: len(x[1]))

            # Create embeddings
            logger.info(f"Creating embeddings for {len(tokenized_corpus)} documents")
            bsize = 256
            # Open file descriptor for the cache file
            with open(cache_file_path, "a") as f:
                for i in tqdm.tqdm(
                    range(0, len(tokenized_corpus), bsize), desc="Creating embeddings"
                ):
                    max_idx = min(i + bsize, len(tokenized_corpus))
                    selected_indices = [
                        tokenized_corpus[idx][0] for idx in range(i, max_idx)
                    ]
                    batch = [tokenized_corpus[idx][1] for idx in range(i, max_idx)]
                    # Pad the batch
                    batch = self.tokenizer.pad(
                        {"input_ids": batch},
                        padding=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    outputs = self.model(
                        batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                    )
                    # Get embeddings
                    embeddings = last_token_pool(
                        outputs.last_hidden_state, batch["attention_mask"]
                    )
                    # Normalize embeddings
                    embeddings = (
                        torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        .cpu()
                        .tolist()
                    )
                    # Write the result to the file
                    for idx, embedding in zip(selected_indices, embeddings):
                        f.write(ujson.dumps([str(idx), embedding]) + "\n")
                    f.flush()

            # Save the index
            logger.info(f"Saving the index to {cache_file_path}")
            f.close()

            # Create the index
            logger.info("Creating the index done!")


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
    retriever = E5MistralRetriever(
        corpus_path=COLLECTION_PATH, skip_loading_model=False
    )

    # Test the retriever
    # print(retriever.retrieve(query))
    # logger.info(retriever.calculate_score(query=query, text=text))
    with slack_utils.notification(
        channel="question-answering",
        success_msg=f"Succeeded to index msmarco with mistral!",
        error_msg=f"Failed to index msmarco with mistral!",
    ):
        retriever.create_index(index_name="msmarco", corpus_path=COLLECTION_PATH)
    logger.info("Done!")
