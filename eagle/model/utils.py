from typing import *

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import segment_coo

from eagle.model.objective import get_target_scale_tensor


def get_valid_num(mask: torch.Tensor) -> torch.Tensor:
    """Get the number of valid tokens for each query
    :param mask with 0 as valid and 1 as non-valid token (Shape: [bsize, num_toks])
    :type mask: torch.Tensor
    :return: num_valid_tokens Shape: [bsize]
    :rtype: torch.Tensor
    """
    num_non_valid_tokens = mask.sum(dim=1)
    target_scale = get_target_scale_tensor(
        target_scale=mask.shape[1],
        b_size=num_non_valid_tokens.shape[0],
        device=num_non_valid_tokens.device,
        dtype=num_non_valid_tokens.dtype,
    )
    num_valid_tokens = target_scale - num_non_valid_tokens
    return num_valid_tokens


def get_scale_factor(mask: torch.Tensor, q_maxlen: int) -> torch.Tensor:
    """Get the scale factor for normalization
    :param mask: Shape: [bsize, num_toks]
    :type mask: torch.Tensor
    :return: scale factor Shape: [bsize]
    :rtype: torch.Tensor
    """
    num_valid_tokens = get_valid_num(mask)
    return q_maxlen / num_valid_tokens


def l1_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1))


def l2_regularization(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(tensor.squeeze(-1)) ** 2


def pid_found_percentage(pids_to_find: List[int], pids_corpus: List[int]) -> float:
    """Calculate the percentage of pids that are found in the corpus"""
    return len(set(pids_to_find).intersection(set(pids_corpus))) / len(pids_to_find)


def initialize_weights(m):
    if isinstance(m, torch.nn.Sequential):
        for layer in m:
            initialize_weights(layer)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def _sort_by_length(ids, mask, bsize=None, descending=False):
    if bsize is not None and ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort(descending=descending).indices
    reverse_indices = indices.sort(descending=False).indices

    return ids[indices], mask[indices], indices, reverse_indices


def _split_into_batches(
    ids: torch.Tensor,
    att_mask: torch.Tensor,
    tok_mask: torch.Tensor = None,
    scatter_indices: List[List[Tuple[int, int]]] = None,
    bsize: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batches = []
    for offset in range(0, ids.size(0), bsize):
        if tok_mask is None and scatter_indices is None:
            batches.append(
                (ids[offset : offset + bsize], att_mask[offset : offset + bsize])
            )
        elif tok_mask is None:
            batches.append(
                (
                    ids[offset : offset + bsize],
                    att_mask[offset : offset + bsize],
                    None,
                    scatter_indices[offset : offset + bsize],
                )
            )
        elif scatter_indices is None:
            batches.append(
                (
                    ids[offset : offset + bsize],
                    att_mask[offset : offset + bsize],
                    tok_mask[offset : offset + bsize],
                    None,
                )
            )
        else:
            batches.append(
                (
                    ids[offset : offset + bsize],
                    att_mask[offset : offset + bsize],
                    tok_mask[offset : offset + bsize],
                    scatter_indices[offset : offset + bsize],
                )
            )

    return batches


def unwrap_logging_items(loss_dic: Dict, target_key: str = None) -> Dict:
    """Remove zero loss items and unwrap torch.Tensor to float

    :param loss_dic: Dictionary containing loss values
    :type loss_dic: Dict
    :return: Dictionary containing loss values with zero values removed
    :rtype: Dict
    """
    if target_key is not None:
        loss_dic = {
            key: value.item() if type(value) == torch.Tensor else value
            for key, value in loss_dic.items()
            if target_key in key
        }
    return {
        key: value
        for key, value in loss_dic.items()
        if value is not None and value != 0
    }


def append_dummy_pid(
    pids: Union[List[int], torch.Tensor], target_pids: List[int], max_num: int
) -> Tuple[Union[List[int], torch.Tensor], List[int]]:
    """Prepare pid lists to evaluate the model with BEIR evaluation methods"""
    is_tensor = isinstance(pids, torch.Tensor)
    if is_tensor:
        dtype = pids.dtype
        device = pids.device
        pids = pids.tolist()

    # Generate dummy pid
    pids_to_append: List[int] = []
    for target_pid in target_pids:
        if target_pid in pids:
            # Find the dummy index that is not in the pids list
            for i in range(100000):
                if i not in pids:
                    pids_to_append.append(i)
                    break
        else:
            pids_to_append.append(target_pid)
    # Generate dummy pids
    for i in range(max_num - len(target_pids)):
        # Find the dummy index that is not in the pids list
        for i in range(100000):
            if i not in pids:
                pids_to_append.append(i)
                break

    # Append the dummy pid to the list
    pids.extend(pids_to_append)

    # Find the index of the target pid in the list
    target_indices: List[int] = [pids.index(target_pid) for target_pid in target_pids]

    # Return the pids with the correct data type
    if is_tensor:
        pids = torch.tensor(pids, dtype=dtype, device=device)
    return pids, target_indices


def has_hf_hook_for_execution_device(module: torch.nn.Module) -> bool:
    if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device"):
        return True
    return False


def modify_execution_device(module: torch.nn.Module, device: int) -> None:
    # Change the execution device
    if has_hf_hook_for_execution_device(module):
        module._hf_hook.execution_device = device
    # Recursively change the execution device
    if isinstance(module, torch.nn.ModuleList):
        for m in module:
            modify_execution_device(m, device)
    for v in dir(module):
        if v == "tile_indices":
            continue
        if module != getattr(module, v) and (
            isinstance(getattr(module, v), torch.nn.Module)
            or isinstance(getattr(module, v), torch.nn.ModuleList)
        ):
            modify_execution_device(getattr(module, v), device)
    return None


def modify_grad(x, inds):
    x[inds] = 0.0
    return x


def aggregate_vectors_with_indices(
    src_tensor: torch.Tensor, scatter_indices: torch.Tensor, reduce: str
) -> torch.Tensor:
    # scatter reduce the tensors
    results = []
    for b_idx, scatter_indices_tensor in enumerate(scatter_indices):
        result = segment_coo(src_tensor[b_idx], scatter_indices_tensor, reduce=reduce)
        results.append(result)
    return pad_sequence(results, batch_first=True, padding_value=0)


def get_weight_layer(
    strategy: str, input_dim: int, intermediate_dim: int, out_dim: int
) -> torch.nn.Module:
    layer = None
    if strategy == "sigmoid":
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, intermediate_dim),
            torch.nn.LayerNorm(intermediate_dim),
            torch.nn.Mish(),
            torch.nn.Linear(intermediate_dim, out_dim),
            torch.nn.Sigmoid(),
        )
    elif strategy == "relu":
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, intermediate_dim),
            torch.nn.LayerNorm(intermediate_dim),
            torch.nn.Mish(),
            torch.nn.Linear(intermediate_dim, out_dim),
            torch.nn.ReLU(),
        )
    elif strategy == "attention":
        # layer = attention_layer
        raise NotImplementedError("Attention layer is not implemented yet.")
    else:
        raise ValueError(f"Unsupported weight strategy: {strategy}")
    return layer


def change_token_weights_with_phrase_information(
    token_weights: torch.Tensor, phrase_scatter_indices: torch.Tensor
) -> torch.Tensor:
    """Change the token weights with the phrase information"""
    bsize = token_weights.shape[0]
    # Check if the shape of the token weights and the phrase scatter indices are the same
    assert token_weights.squeeze().shape == phrase_scatter_indices.squeeze().shape, (
        f"The shape of the token weights and the phrase scatter indices must be the same"
        f"Token weights shape: {token_weights.squeeze().shape}, "
        f"Phrase scatter indices shape: {phrase_scatter_indices.squeeze().shape}"
    )
    new_token_weights = [[] for _ in range(bsize)]
    for b_idx, phrase_scatter_indices_tensor in enumerate(phrase_scatter_indices):
        tmp = []
        for i, idx in enumerate(phrase_scatter_indices_tensor):
            weight = token_weights[b_idx][idx]
            if i == 0:
                tmp.append(weight)
            else:
                # Check if the current index is the same as the previous index
                # if it is, then add the token weight to the tmp list
                # if it is not, then rescale the token weights and save in the final list. Then add the current index to the tmp list
                if weight == phrase_scatter_indices_tensor[i - 1]:
                    tmp.append(weight)
                else:
                    # Rescale the token weights and save in the final list
                    max_value = min(tmp)
                    new_token_weights[b_idx].extend([max_value] * len(tmp))
                    tmp = [weight]
        # Save the left over token weights
        if len(tmp) > 0:
            max_value = max(tmp)
            new_token_weights[b_idx].extend([max_value] * len(tmp))

    # Convert the list of lists to a tensor
    new_token_weights = torch.tensor(
        new_token_weights, dtype=token_weights.dtype, device=token_weights.device
    ).unsqueeze(-1)

    return new_token_weights
