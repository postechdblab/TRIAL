from typing import *

import spacy
import hkkang_utils.pattern as pattern_utils


class Sentencizer(metaclass=pattern_utils.SingletonMetaWithArgs):
    def __init__(self, gpu_id: int = 0):
        model_name = "en_core_web_trf"
        if gpu_id != -1:
            spacy.require_gpu(gpu_id=gpu_id)
        try:
            self.spacy_model = spacy.load(model_name)
        except:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.spacy_model = spacy.load(model_name)
        self.spacy_model.add_pipe("sentencizer")

    def __call__(self, text_or_texts: Union[str, List[str]]):
        if isinstance(text_or_texts, str):
            return self.split_into_sentences(text_or_texts)
        elif isinstance(text_or_texts, list):
            return self.split_into_sentences_batch(text_or_texts)
        else:
            raise ValueError(f"Invalid input type: {type(text_or_texts)})")

    def split_into_sentences(self, text: str) -> List[str]:
        return self.split_into_sentences_batch([text])[0]

    def split_into_sentences_batch(self, texts: List[str]) -> List[List[str]]:
        spacy_results = self.spacy_model.pipe(texts)
        all_sentences: List[List[str]] = []
        for i, sentences in enumerate(spacy_results):
            sentences: List[str] = [sent.text for sent in sentences.sents]
            all_sentences.append(sentences)
        return all_sentences
