from typing import *

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import segment_coo


def initialize_weights(m):
    if isinstance(m, torch.nn.Sequential):
        for layer in m:
            initialize_weights(layer)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, att_mask, tok_mask=None, bsize: int = 1):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        if tok_mask is None:
            batches.append(
                (ids[offset : offset + bsize], att_mask[offset : offset + bsize])
            )
        else:
            batches.append(
                (
                    ids[offset : offset + bsize],
                    att_mask[offset : offset + bsize],
                    tok_mask[offset : offset + bsize],
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
    pids: Union[List[int], torch.Tensor], target_pids: List[int]
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


def get_vectors_from_ranges(
    tensors: torch.Tensor, scatter_indices: torch.Tensor, reduce: str
) -> torch.Tensor:
    # scatter reduce the tensors
    results = []
    for b_idx, scatter_indices_tensor in enumerate(scatter_indices):
        result = segment_coo(tensors[b_idx], scatter_indices_tensor, reduce=reduce)
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
