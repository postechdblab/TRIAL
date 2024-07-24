import os

import torch
import tqdm
import ujson

from eagle.index.codecs.residual import ResidualCodec
from eagle.index.utils import optimize_ivf
from eagle.search.strided_tensor import StridedTensor


class IndexLoader:
    def __init__(self, index_path, use_gpu=True, load_index_with_mmap=False):
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.load_index_with_mmap = load_index_with_mmap

        self._load_codec()  # Centroids information

        # Load ivfs
        self.cls_ivf = self._load_ivf(granularity="cls", must_exists=False)
        self.tok_ivf = self._load_ivf(granularity="tok", must_exists=True)
        self.phrase_ivf = self._load_ivf(granularity="phrase", must_exists=False)

        self.cls_lens = self._load_item_lens(granularity="cls", must_exists=False)
        self.tok_lens = self._load_item_lens(granularity="tok", must_exists=True)
        self.phrase_lens = self._load_item_lens(
            granularity="phrase", must_exists=False
        )  # Document lengths
        self._load_embeddings()  # Document embeddings

    def _load_codec(self):
        print(f"#> Loading codec...")
        self.codec = ResidualCodec.load(self.index_path)

    def _load_ivf(self, granularity: str, must_exists: bool = True) -> StridedTensor:
        print(f"#> Loading IVF...")

        ivf_path = os.path.join(self.index_path, f"{granularity}-ivf.pt")
        ivf_pid_path = os.path.join(self.index_path, f"{granularity}-ivf.pid.pt")
        if os.path.exists(ivf_pid_path):
            ivf, ivf_lengths = torch.load(ivf_pid_path, map_location="cpu")
        else:
            if not must_exists:
                return None
            assert os.path.exists(ivf_path)
            ivf, ivf_lengths = torch.load(ivf_path, map_location="cpu")
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        # ivf, ivf_lengths = ivf.cuda(), torch.LongTensor(ivf_lengths).cuda()  # FIXME: REMOVE THIS LINE!
        ivf = StridedTensor(ivf, ivf_lengths, use_gpu=self.use_gpu)

        return ivf

    def _load_item_lens(
        self, granularity: str, must_exists: bool = True
    ) -> torch.Tensor:
        doclens = []

        print("#> Loading doclens...")

        for chunk_idx in tqdm.tqdm(range(self.num_chunks)):
            file_path = os.path.join(
                self.index_path, f"{granularity}_lens.{chunk_idx}.json"
            )
            # Skip if file doesn't exist
            if not must_exists and not os.path.exists(file_path):
                continue

            with open(file_path) as f:
                chunk_doclens = ujson.load(f)
                doclens.extend(chunk_doclens)

        return torch.tensor(doclens)

    def _load_embeddings(self) -> None:
        cls_embeddings, tok_embeddings, phrase_embeddings = (
            ResidualCodec.Embeddings.load_chunks(
                index_path=self.index_path,
                chunk_idxs=range(self.num_chunks),
                num_cls_embeddings=self.num_cls_embeddings,
                num_tok_embeddings=self.num_tok_embeddings,
                num_phrase_embeddings=self.num_phrase_embeddings,
                load_index_with_mmap=self.load_index_with_mmap,
            )
        )
        self.cls_embeddings = cls_embeddings
        self.tok_embeddings = tok_embeddings
        self.phrase_embeddings = phrase_embeddings

    @property
    def metadata(self):
        try:
            self._metadata
        except:
            with open(os.path.join(self.index_path, "metadata.json")) as f:
                self._metadata = ujson.load(f)

        return self._metadata

    @property
    def config(self):
        raise NotImplementedError()  # load from dict at metadata['config']

    @property
    def num_chunks(self):
        # EVENTUALLY: If num_chunks doesn't exist (i.e., old index), fall back to counting doclens.*.json files.
        return self.metadata["num_chunks"]

    @property
    def num_cls_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata["num_cls_embeddings"]

    @property
    def num_tok_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata["num_tok_embeddings"]

    @property
    def num_phrase_embeddings(self):
        # EVENTUALLY: If num_embeddings doesn't exist (i.e., old index), sum the values in doclens.*.json files.
        return self.metadata["num_phrase_embeddings"]
