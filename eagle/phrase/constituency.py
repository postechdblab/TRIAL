import functools
import logging
import string
from typing import *

import benepar
import hkkang_utils.data as data_utils
import hkkang_utils.pattern as pattern_utils
import spacy
import tqdm

MAX_TOKEN_LENGTH = 512

logger = logging.getLogger("Constituency")
# nlp = spacy.load(
#     "en_core_web_sm",
#     disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
# )
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


@spacy.language.Language.component("limit_token_length")
# Create a custom component to filter out long tokens
def limit_token_length(doc, model=None) -> None:
    # Create a list of filtered tokens and their sentence starts
    if len(doc) >= MAX_TOKEN_LENGTH:
        # Create new Doc object with filtered tokens
        last_token_idx = MAX_TOKEN_LENGTH - 1
        # last_token_idx = len(doc) - 1
        last_char_idx = doc[last_token_idx].idx + len(doc[last_token_idx].text)
        filtered_text = doc.text[:last_char_idx]
        # Create a new Doc object by performing tokenization again
        doc = nlp(filtered_text)
    return doc


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
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.model = spacy.load(model_name)
        self.model.add_pipe("limit_token_length")
        try:
            self.model.add_pipe(
                "benepar", config={"model": "benepar_en3", "subbatch_max_tokens": 5000}
            )
        except:
            import nltk

            benepar.download("benepar_en3")
            self.model.add_pipe(
                "benepar", config={"model": "benepar_en3", "subbatch_max_tokens": 5000}
            )

    def __call__(
        self,
        texts: List[str],
        show_progress: bool = False,
        batch_size: int = 10000,
    ) -> List[List[Phrase]]:
        # Parse the texts in batches
        parsed_results: List = []

        for doc_batch in tqdm.tqdm(
            self.model.pipe(texts, batch_size=batch_size), disable=not show_progress
        ):
            parsed_results.append(doc_batch)
        # Enumerate the results
        all_doc_phrases: List[List[Phrase]] = []
        for item in parsed_results:
            doc_phrases: List[Phrase] = []
            # Enumerate over the sentences
            for s_idx, sent in enumerate(item.sents):
                phrases_in_sent, _ = self.traverse(sent)
                doc_phrases.extend(phrases_in_sent)
            all_doc_phrases.append(doc_phrases)
        return all_doc_phrases

    def traverse(self, tree: spacy.tokens.span.Span) -> Tuple[List, bool]:
        """Perform post-order traversal of the constituency tree.

        :param tree: Constituency tree
        :type tree: spacy.tokens.span.Span
        :return: List of phrases and whether to concatenate the phrases with the parent?
        :rtype: Tuple[List, bool]
        """
        if len(list(tree._.children)) == 0:
            return [Phrase(tree.text, tree.start_char)], False
        return_list = []
        label = tree._.labels[0]
        is_noun_phrase = label == "NP"
        has_noun_phrase_child = False
        for p_idx, span in enumerate(tree._.children):
            # Go deeper
            child_phrases, is_child_noun_phrase = self.traverse(span)
            has_noun_phrase_child = has_noun_phrase_child or is_child_noun_phrase
            return_list.extend(child_phrases)
        # Combine the children if 1) the current node is noun phrase and 2) there is no noun phrase child
        if is_noun_phrase and not has_noun_phrase_child:
            return_list = [
                Phrase(
                    " ".join([p.text for p in return_list]), return_list[0].start_idx
                )
            ]
        contains_noun_phrase = is_noun_phrase or has_noun_phrase_child

        return return_list, contains_noun_phrase
