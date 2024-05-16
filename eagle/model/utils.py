import functools
from typing import *

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import segment_coo


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
            torch.nn.ReLU(),
            torch.nn.Linear(intermediate_dim, out_dim),
            torch.nn.Sigmoid(),
        )
    elif strategy == "relu":
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, intermediate_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(intermediate_dim, out_dim),
            torch.nn.ReLU(),
        )
    elif strategy == "attention":
        # layer = attention_layer
        raise NotImplementedError("Attention layer is not implemented yet.")
    else:
        raise ValueError(f"Unsupported weight strategy: {strategy}")
    return layer


@functools.lru_cache(maxsize=32)
def get_ib_loss_label(q_n: int, nway: int, device: torch.dtype) -> torch.Tensor:
    return torch.arange(0, q_n, device=device, requires_grad=False) * nway


@functools.lru_cache(maxsize=32)
def get_loss_label(b_size: int, device: torch.dtype) -> torch.Tensor:
    return torch.zeros(b_size, dtype=torch.long, device=device, requires_grad=False)


@functools.lru_cache(maxsize=1)
def indices_to_select_docs_for_ib_loss(q_n: int, nway: int) -> List[int]:
    all_docs_num_for_q = nway * q_n
    indices = []
    for j in range(q_n):
        offset = all_docs_num_for_q * j
        # Add neg docs from it's previous queries
        tmps = []
        for k in range(0, j):
            start_idx = k * nway
            tmps.extend(list(range(start_idx, start_idx + nway)))
        # Add gold doc from it's query
        tmps.append(j * nway)
        # Add neg docs from it's next queries
        for k in range(j + 1, q_n):
            start_idx = k * nway
            tmps.extend(list(range(start_idx, start_idx + nway)))
        # Add offset
        tmps = [tmp + offset for tmp in tmps]
        indices.extend(tmps)
    return indices


@functools.lru_cache(maxsize=1)
def doc_indices_for_ib_loss(q_n: int, nway: int, nhard: int) -> List[int]:
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
    return indices
