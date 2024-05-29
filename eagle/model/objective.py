import functools
from typing import *

import torch

from eagle.model.compiled_tensor_op import compute_loss_c


@functools.lru_cache(maxsize=32)
def get_target_scale_tensor(
    target_scale: int, b_size: int, device: torch.dtype, dtype: torch.dtype
) -> torch.Tensor:
    return torch.full(
        (b_size, 1),
        fill_value=target_scale,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


@functools.lru_cache(maxsize=32)
def get_ib_loss_label(q_n: int, nway: int, device: torch.dtype) -> torch.Tensor:
    return torch.arange(0, q_n, device=device, requires_grad=False) * nway


@functools.lru_cache(maxsize=32)
def get_loss_label(b_size: int, device: torch.dtype) -> torch.Tensor:
    return torch.zeros(b_size, dtype=torch.long, device=device, requires_grad=False)


@functools.lru_cache(maxsize=1000)
def doc_indices_for_ib_loss(
    q_n: int, nway: int, nhard: int, return_as_tensor: bool = False, device=None
) -> Union[List[int], torch.Tensor]:
    indices = []
    for j in range(q_n):
        tmps = []
        for k in range(0, j):
            start_idx = k * nway
            tmps.extend(list(range(start_idx, start_idx + nhard)))
        # Add gold doc from it's query
        tmps.append(j * nway)
        # Add neg docs from it's next queries
        for k in range(j + 1, q_n):
            start_idx = k * nway
            tmps.extend(list(range(start_idx, start_idx + nhard)))
        # Add offset
        indices.extend(tmps)
    if return_as_tensor:
        indices = torch.tensor(
            indices, dtype=torch.long, requires_grad=False, device=device
        )
    return indices


def compute_loss(
    scores: torch.Tensor,
    ib_scores: torch.Tensor,
    distillation_scores: torch.Tensor,
    bsize: int,
    nway: int,
    ib_nhard: int,
    device: torch.device,
    intra_loss_coeff: Optional[float] = None,
    inter_loss_coeff: Optional[float] = None,
    distillation_loss_coeff: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = get_loss_label(bsize, device=device)
    ib_labels = get_ib_loss_label(bsize, ib_nhard, device=device)
    return compute_loss_c(
        scores=scores, 
        ib_scores=ib_scores, 
        labels=labels, 
        ib_labels=ib_labels, 
        nway=nway, 
        ce_loss_coeff=intra_loss_coeff, 
        ib_loss_coeff=inter_loss_coeff,
        kl_loss_coeff=distillation_loss_coeff,
        distillation_scores=distillation_scores
    )


def compute_fine_grained_loss(
    scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor]:
    # ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')(scores, labels)
    ce_loss = torch.nn.CrossEntropyLoss()(scores, labels)
    return ce_loss
