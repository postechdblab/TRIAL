from typing import *

import torch

CAPABILITY = torch.cuda.get_device_capability()


# @torch.compile(dynamic=True, fullgraph=True, mode="max-autotune")
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


def l1_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1))


def l2_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1)) ** 2


# # Compile the functions
# if CAPABILITY[0] >= 7:
#     compute_loss_c = torch.compile(compute_loss_c, dynamic=True, fullgraph=True, mode="max-autotune")
