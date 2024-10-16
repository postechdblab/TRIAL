import functools
from typing import *

import torch


@functools.lru_cache(maxsize=32)
def get_target_scale_tensor(
    target_scale: int, b_size: int, device: torch.dtype, dtype: torch.dtype
) -> torch.Tensor:
    return torch.full(
        (b_size,),
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
        distillation_scores=distillation_scores,
    )


def compute_loss_c(
    scores: torch.Tensor,
    ib_scores: torch.Tensor,
    labels,
    ib_labels,
    nway: int,
    ce_loss_coeff: Optional[float],
    ib_loss_coeff: Optional[float],
    kl_loss_coeff: Optional[float],
    distillation_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Compute inter-data cross-entropy loss
    scores = scores.view(-1, nway)
    q_n = scores.size(0)
    # Compute inter-data cross-entropy loss (i.e., in-batch negatives)
    ib_loss = torch.nn.CrossEntropyLoss()(ib_scores.view(q_n, -1), ib_labels)
    if ib_loss_coeff is not None and ib_loss_coeff != 1:
        ib_loss = ib_loss_coeff * ib_loss

    ce_loss = None
    kl_loss = None
    if distillation_scores is not None:
        # Compute KL divergence loss if doing knowledge distillation
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
            log_scores, distillation_scores
        )
        if kl_loss_coeff is not None and kl_loss_coeff != 1:
            kl_loss = kl_loss_coeff * kl_loss
    else:
        # Compute intra cross-entropy loss
        ce_loss = torch.nn.CrossEntropyLoss()(scores, labels)
        if ce_loss_coeff is not None and ce_loss_coeff != 1:
            ce_loss = ce_loss_coeff * ce_loss

    # Aggregate the losses
    loss = ib_loss
    if ce_loss is not None:
        loss = loss + ce_loss

    if kl_loss is not None:
        loss = loss + kl_loss

    return loss, ce_loss, ib_loss, kl_loss


def compute_fine_grained_loss(
    scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor]:
    # ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')(scores, labels)
    ce_loss = torch.nn.CrossEntropyLoss()(scores, labels)
    return ce_loss
