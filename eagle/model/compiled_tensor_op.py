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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Compute inter-data cross-entropy loss
    scores = scores.view(-1, nway)
    q_n = scores.size(0)

    ce_loss = torch.nn.CrossEntropyLoss()(scores, labels)

    # Compute intra-data cross-entropy loss (i.e., in-batch negatives)
    ib_loss = torch.nn.CrossEntropyLoss()(ib_scores.view(q_n, -1), ib_labels)

    if ce_loss_coeff is not None:
        ce_loss = ce_loss_coeff * ce_loss

    if ib_loss_coeff is not None:
        ib_loss = ib_loss_coeff * ib_loss

    loss = ce_loss + ib_loss

    return loss, ce_loss, ib_loss


def l1_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1))


def l2_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1)) ** 2


# # Compile the functions
# if CAPABILITY[0] >= 7:
#     compute_loss_c = torch.compile(compute_loss_c, dynamic=True, fullgraph=True, mode="max-autotune")
