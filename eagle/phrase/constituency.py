import functools
import logging
import string
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.list as list_utils
import hkkang_utils.pattern as pattern_utils
import spacy
import tqdm

logger = logging.getLogger("Constituency")


@data_utils.dataclass
class Phrase:
    text: str
    start_idx: int

    @property
    def end_idx(self) -> int:
        return self.start_idx + len(self.text)

    @property
    def idx_range(self) -> Tuple[int, int]:
        return self.start_idx, self.end_idx


class ConstituencyParser(metaclass=pattern_utils.SingletonMetaWithArgs):
    def __init__(self, gpu_id: int = 0) -> None:
        if gpu_id != -1:
            spacy.require_gpu(gpu_id=gpu_id)
        model_name = "en_core_web_sm"
        try:
            self.model = spacy.load(
                model_name,
                disable=[
                    "tok2vec",
                    "tagger",
                    # "parser",
                    "attribute_ruler",
                    "lemmatizer",
                    "ner",
                ],
            )
            # self.model.add_pipe("senter")
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.model = spacy.load(model_name)
        self.model.add_pipe("benepar", config={"model": "benepar_en3"})

    def __call__(
        self,
        texts: List[str],
        show_progress: bool = False,
        batch_size: int = 1000,
    ) -> List[List[Phrase]]:
        # Parse the texts in batches
        parsed_results: List = []
        import time

        t1 = time.time()
        for doc_batch in tqdm.tqdm(
            self.model.pipe(texts, batch_size=batch_size), disable=not show_progress
        ):
            parsed_results.append(doc_batch)
        print(f"Time taken: {time.time() - t1}")
        # Enumerate the results
        all_phrases: List[List[Phrase]] = []
        for item in parsed_results:
            phrases: List[Phrase] = []
            # Enumerate over the sentences
            for s_idx, sent in enumerate(item.sents):
                phrases.extend(self.traverse(sent))
            all_phrases.append(phrases)
        return all_phrases

    def traverse(self, tree: spacy.tokens.span.Span, depth: int = 0) -> List:
        return_list = []
        for p_idx, span in enumerate(tree._.children):
            # Check if need to go deeper
            text_from_noun_chunks = " ".join(nc.text for nc in span.noun_chunks)
            go_deeper = text_from_noun_chunks and (text_from_noun_chunks != span.text)
            if go_deeper:
                return_list.extend(self.traverse(span, depth + 1))
            else:
                return_list.append(Phrase(span.text, span.start_char))
        return return_list
