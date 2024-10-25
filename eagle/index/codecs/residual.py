import os
import pathlib
from itertools import product
from typing import *

import hkkang_utils.file as file_utils
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.cpp_extension import load

from eagle.index.codecs.residual_embeddings import BaseResidualEmbeddings


class ResidualCodec:
    Embeddings = BaseResidualEmbeddings

    def __init__(
        self,
        cfg: DictConfig,
        centroids: torch.Tensor = None,
        avg_residual=None,
        bucket_cutoffs=None,
        bucket_weights=None,
    ):
        # Load the torch extensions
        ResidualCodec.try_load_torch_extensions(self.use_gpu)

        # Save configs
        self.cfg = cfg
        self.dim = cfg.dim
        self.nbits = cfg.nbits

        # Convert dtype if necessary
        if self.use_gpu:
            centroids = centroids.half().cuda()
            if torch.is_tensor(avg_residual):
                avg_residual = avg_residual.half().cuda()

            if torch.is_tensor(bucket_cutoffs):
                bucket_cutoffs = bucket_cutoffs.cuda()
                bucket_weights = bucket_weights.half().cuda()
        else:
            centroids = centroids.float()
            if torch.is_tensor(avg_residual):
                avg_residual = avg_residual.to(torch.float32)

            if torch.is_tensor(bucket_weights):
                bucket_weights = bucket_weights.to(torch.float32)

        # Save arguments
        self.centroids = centroids
        self.avg_residual = avg_residual
        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights
        # Initialize the reversed bit map and the decompression lookup table
        self.reversed_bit_map = self._init_reversed_bit_map()
        self.decompression_lookup_table = self._init_decompression_lookup_table(
            bucket_weights=self.bucket_weights
        )

    @property
    def use_gpu(self) -> bool:
        return torch.cuda.is_available()

    @property
    def arange_bits(self) -> torch.Tensor:
        return torch.arange(0, self.nbits, device=self.device, dtype=torch.uint8)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if self.use_gpu else "cpu")

    @property
    def keys_per_byte(self) -> int:
        return 8 // self.nbits

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or not use_gpu:
            return

        print(
            f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "kernel/decompress_residuals.cpp",
                ),
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "kernel/decompress_residuals.cu",
                ),
            ],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        print(
            f"Loading packbits_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        packbits_cpp = load(
            name="packbits_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "kernel/packbits.cpp"
                ),
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "kernel/packbits.cu"
                ),
            ],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.packbits = packbits_cpp.packbits_cpp

        cls.loaded_extensions = True

    @classmethod
    def load(cls, index_path: str) -> "ResidualCodec":
        plan_path = os.path.join(index_path, "plan.json")
        centroids_path = os.path.join(index_path, "centroids.pt")
        avgresidual_path = os.path.join(index_path, "avg_residual.pt")
        buckets_path = os.path.join(index_path, "buckets.pt")

        plan = OmegaConf.create(file_utils.read_json_file(plan_path))
        centroids = torch.load(centroids_path, map_location="cpu", weights_only=True)
        avg_residual = torch.load(
            avgresidual_path, map_location="cpu", weights_only=True
        )
        bucket_cutoffs, bucket_weights = torch.load(
            buckets_path, map_location="cpu", weights_only=True
        )

        if avg_residual.dim() == 0:
            avg_residual = avg_residual.item()

        return cls(
            cfg=plan.config,
            centroids=centroids,
            avg_residual=avg_residual,
            bucket_cutoffs=bucket_cutoffs,
            bucket_weights=bucket_weights,
        )

    def _init_reversed_bit_map(self) -> torch.Tensor:
        # We reverse the residual bits because arange_bits as
        # currently constructed produces results with the reverse
        # of the expected endianness
        reversed_bit_map: List[int] = []
        mask = (1 << self.nbits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -self.nbits):
                # Extract a subsequence of length n bits
                x = (i >> (j - self.nbits)) & mask

                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(self.nbits - 1, -1, -1):
                    y += ((x >> (self.nbits - k - 1)) & 1) * (2**k)

                # Set the corresponding bits in the output byte
                z |= y
                if j > self.nbits:
                    z <<= self.nbits
            reversed_bit_map.append(z)
        reversed_bit_map = torch.tensor(reversed_bit_map).to(torch.uint8)

        if self.use_gpu:
            reversed_bit_map = reversed_bit_map.cuda()

        return reversed_bit_map

    def _init_decompression_lookup_table(
        self, bucket_weights: Optional[torch.Tensor]
    ) -> Union[None, torch.Tensor]:
        # A table of all possible lookup orders into bucket_weights
        # given n bits per lookup
        if bucket_weights is None:
            return None
        decomporession_lookup_table = torch.tensor(
            list(
                product(
                    list(range(len(self.bucket_weights))),
                    repeat=self.keys_per_byte,
                )
            ),
            device=self.device,
            dtype=torch.uint8,
        )
        return decomporession_lookup_table

    def save(self, index_path: str) -> None:
        assert self.avg_residual is not None
        assert torch.is_tensor(self.bucket_cutoffs), self.bucket_cutoffs
        assert torch.is_tensor(self.bucket_weights), self.bucket_weights

        centroids_path = os.path.join(index_path, "centroids.pt")
        avgresidual_path = os.path.join(index_path, "avg_residual.pt")
        buckets_path = os.path.join(index_path, "buckets.pt")

        torch.save(self.centroids, centroids_path)
        torch.save((self.bucket_cutoffs, self.bucket_weights), buckets_path)

        if torch.is_tensor(self.avg_residual):
            torch.save(self.avg_residual, avgresidual_path)
        else:
            torch.save(torch.tensor([self.avg_residual]), avgresidual_path)

    def compress(self, embs: torch.Tensor) -> BaseResidualEmbeddings:
        codes, residuals = [], []

        for batch in embs.split(1 << 18):
            if self.use_gpu:
                batch = batch.cuda()
            codes_ = self.compress_into_codes(batch, out_device=batch.device)
            centroids_ = self.lookup_centroids(codes_, out_device=batch.device)

            residuals_ = batch - centroids_

            codes.append(codes_.cpu())
            residuals.append(self.binarize(residuals_).cpu())

        codes = torch.cat(codes)
        residuals = torch.cat(residuals)

        return ResidualCodec.Embeddings(codes, residuals)

    def binarize(self, residuals) -> torch.Tensor:
        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(
            dtype=torch.uint8
        )
        residuals = residuals.unsqueeze(-1).expand(
            *residuals.size(), self.nbits
        )  # add a new nbits-wide dim
        residuals = (
            residuals >> self.arange_bits
        )  # divide by 2^bit for each bit position
        residuals = residuals & 1  # apply mod 2 to binarize

        assert self.dim % 8 == 0
        assert self.dim % (self.nbits * 8) == 0, (self.dim, self.nbits)

        if self.use_gpu:
            residuals_packed = ResidualCodec.packbits(residuals.contiguous().flatten())
        else:
            residuals_packed = np.packbits(np.asarray(residuals.contiguous().flatten()))
        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)
        residuals_packed = residuals_packed.reshape(
            residuals.size(0), self.dim // 8 * self.nbits
        )

        return residuals_packed

    def compress_into_codes(self, embs, out_device: torch.device) -> torch.Tensor:
        """
        EVENTUALLY: Fusing the kernels or otherwise avoiding materalizing the entire matrix before max(dim=0)
                    seems like it would help here a lot.
        """

        codes = []

        bsize = (1 << 29) // self.centroids.size(0)
        for batch in embs.split(bsize):
            if self.use_gpu:
                indices = (
                    (self.centroids @ batch.T.cuda().half())
                    .max(dim=0)
                    .indices.to(device=out_device)
                )
            else:
                indices = (
                    (self.centroids @ batch.T.cpu().float())
                    .max(dim=0)
                    .indices.to(device=out_device)
                )
            codes.append(indices)

        return torch.cat(codes)

    def lookup_centroids(self, codes, out_device: torch.device) -> torch.Tensor:
        """
        Handles multi-dimensional codes too.

        EVENTUALLY: The .split() below should happen on a flat view.
        """

        centroids = []

        for batch in codes.split(1 << 20):
            if self.use_gpu:
                centroids.append(
                    self.centroids[batch.cuda().long()].to(device=out_device)
                )
            else:
                centroids.append(self.centroids[batch.long()].to(device=out_device))

        return torch.cat(centroids)

    # @profile
    def decompress(self, compressed_embs: BaseResidualEmbeddings) -> torch.Tensor:
        """
        We batch below even if the target device is CUDA to avoid large temporary buffers causing OOM.
        """

        codes, residuals = compressed_embs.codes, compressed_embs.residuals

        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_, residuals_ = codes_.cuda(), residuals_.cuda()
                centroids_ = ResidualCodec.decompress_residuals(
                    residuals_,
                    self.bucket_weights,
                    self.reversed_bit_map,
                    self.decompression_lookup_table,
                    codes_,
                    self.centroids.half(),
                    self.dim,
                    self.nbits,
                ).cuda()
            else:
                # TODO: Remove dead code
                centroids_ = self.lookup_centroids(codes_, out_device="cpu")
                residuals_ = self.reversed_bit_map[residuals_.long()]
                residuals_ = self.decompression_lookup_table[residuals_.long()]
                residuals_ = residuals_.reshape(residuals_.shape[0], -1)
                residuals_ = self.bucket_weights[residuals_.long()]
                centroids_.add_(residuals_)

            if self.use_gpu:
                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            else:
                D_ = torch.nn.functional.normalize(
                    centroids_.to(torch.float32), p=2, dim=-1
                )
            D.append(D_)

        return torch.cat(D)
