import os
import queue
import threading
from contextlib import contextmanager
from typing import *

import torch
import ujson
from omegaconf import DictConfig

from eagle.index.codecs.residual import ResidualCodec
from eagle.index.codecs.residual_embeddings import ResidualEmbeddings


class IndexSaver:
    def __init__(self, cfg: DictConfig, dir_path: str) -> None:
        self.cfg = cfg
        self.dir_path = dir_path

    def _saver_thread(self) -> None:
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _write_chunk_to_disk(
        self,
        chunk_idx: int,
        offset: int,
        compressed_cls_embs: ResidualEmbeddings,
        compressed_tok_embs: ResidualEmbeddings,
        compressed_phrase_embs: ResidualEmbeddings,
        tok_lens: List[int],
        phrase_lens: List[int],
    ) -> None:
        path_prefix = os.path.join(self.dir_path, str(chunk_idx))
        if compressed_cls_embs is not None:
            compressed_cls_embs.save(path_prefix + "-cls")
        compressed_tok_embs.save(path_prefix + "-tok")
        if compressed_phrase_embs is not None:
            compressed_phrase_embs.save(path_prefix + "-phrase")

        cls_lens_path = os.path.join(self.dir_path, f"cls_lens.{chunk_idx}.json")
        tok_lens_path = os.path.join(self.dir_path, f"tok_lens.{chunk_idx}.json")
        phrase_lens_path = os.path.join(self.dir_path, f"phrase_lens.{chunk_idx}.json")

        # Save the lengths of the embeddings
        if compressed_cls_embs is not None:
            with open(cls_lens_path, "w") as output_cls_lens:
                cls_lens = [1] * compressed_cls_embs.codes.size(0)
                ujson.dump(cls_lens, output_cls_lens)
        with open(tok_lens_path, "w") as output_tok_lens:
            ujson.dump(tok_lens, output_tok_lens)
        if compressed_phrase_embs is not None:
            with open(phrase_lens_path, "w") as output_phrase_lens:
                ujson.dump(phrase_lens, output_phrase_lens)

        metadata_path = os.path.join(self.dir_path, f"{chunk_idx}.metadata.json")
        with open(metadata_path, "w") as output_metadata:
            metadata = {
                "passage_offset": offset,
                "num_passages": len(tok_lens),
                "num_cls_embeddings": (
                    len(compressed_cls_embs) if compressed_cls_embs is not None else 0
                ),
                "num_tok_embeddings": len(compressed_tok_embs),
                "num_phrase_embeddings": (
                    len(compressed_phrase_embs)
                    if compressed_phrase_embs is not None
                    else 0
                ),
            }
            ujson.dump(metadata, output_metadata)

    def save_codec(self, codec: ResidualCodec) -> None:
        codec.save(index_path=self.dir_path)

    def load_codec(self):
        return ResidualCodec.load(index_path=self.dir_path)

    def try_load_codec(self) -> bool:
        try:
            ResidualCodec.load(index_path=self.dir_path)
            return True
        except Exception as e:
            return False

    def check_chunk_exists(self, chunk_idx):
        # TODO: Verify that the chunk has the right amount of data?

        tok_lens_path = os.path.join(self.dir_path, f"tok_lens.{chunk_idx}.json")
        if not os.path.exists(tok_lens_path):
            return False

        metadata_path = os.path.join(self.dir_path, f"{chunk_idx}.metadata.json")
        if not os.path.exists(metadata_path):
            return False

        path_prefix = os.path.join(self.dir_path, str(chunk_idx))
        codes_path = f"{path_prefix}-tok.codes.pt"
        if not os.path.exists(codes_path):
            return False

        residuals_path = (
            f"{path_prefix}-tok.residuals.pt"  # f'{path_prefix}.residuals.bn'
        )
        if not os.path.exists(residuals_path):
            return False

        return True

    @contextmanager
    def thread(self) -> Generator:
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(
        self,
        chunk_idx: int,
        offset: int,
        cls_embs: Optional[torch.Tensor],
        tok_embs: torch.Tensor,
        phrase_embs: Optional[torch.Tensor],
        tok_lens: List[int],
        phrase_lens: List[int],
    ) -> None:
        compressed_tok_embs = self.codec.compress(tok_embs)

        if cls_embs is None:
            compressed_cls_embs = None
        else:
            compressed_cls_embs = self.codec.compress(cls_embs)

        if phrase_embs is None:
            compressed_phrase_embs = None
        else:
            compressed_phrase_embs = self.codec.compress(phrase_embs)

        self.saver_queue.put(
            (
                chunk_idx,
                offset,
                compressed_cls_embs,
                compressed_tok_embs,
                compressed_phrase_embs,
                tok_lens,
                phrase_lens,
            )
        )
