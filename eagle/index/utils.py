import os
import re
from typing import *

import hkkang_utils.list as list_utils
import torch
import tqdm
import ujson


def flatten_items_with_mask(
    items: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, List[int]]:
    """Flatten the given items tensor based on the given mask.
    :param items: Shape: (A, B, dim) or (A, dim)
    :type items: torch.Tensor
    :param mask: Shape (A, B)
    :type mask: torch.Tensor
    :return: Flattened items of shape (C, dim), and lengths of each item in the flattened tensor
    :rtype: Tuple[torch.Tensor, List[int]]
    """
    assert (
        items.shape[:2] == mask.shape
    ), f"Shape mismatch: {items.shape} vs {mask.shape}"
    assert mask.dtype == torch.bool, f"Type mismatch: {mask.dtype}"
    assert items.dim() in [2, 3], f"Expected 2D or 3D tensor, got {items.dim()}D tensor"

    # Reverse the mask
    mask = ~mask
    lengths = mask.sum(dim=1).tolist()
    items = items[mask]

    return items, lengths


def all_gather_nd(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        (Tensor): output list of tensors that can be of different sizes
    """
    # Gather the size of the tensors
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)

    # Find the max size
    max_length = max(size[0] for size in all_sizes)

    # Create a tensor of max size and fill it with the current tensor and padding
    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    # Gather all tensors
    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(all_tensors_padded, tensor)

    # Remove the padding
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[: size[0]])
    return all_tensors


def load_item_lens(directory, flatten=True, granularity="tok"):
    doclens_filenames = {}

    for filename in os.listdir(directory):
        match = re.match(f"{granularity}_lens.(\d+).json", filename)

        if match is not None:
            doclens_filenames[int(match.group(1))] = filename

    doclens_filenames = [
        os.path.join(directory, doclens_filenames[i])
        for i in sorted(doclens_filenames.keys())
    ]

    all_doclens = [ujson.load(open(filename)) for filename in doclens_filenames]

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    if len(all_doclens) == 0:
        raise ValueError("Could not load doclens")

    return all_doclens


def optimize_ivf(
    orig_ivf, orig_ivf_lengths, index_path, verbose: int = 3, granularity: str = "tok"
):
    if verbose > 1:
        print("#> Optimizing IVF to store map from centroids to list of pids..")

        print("#> Building the emb2pid mapping..")
    all_item_lens = load_item_lens(index_path, flatten=False, granularity=granularity)

    # assert self.num_embeddings == sum(flatten(all_doclens))

    all_item_lens = list_utils.do_flatten_list(all_item_lens)
    total_num_tok_embeddings = sum(all_item_lens)

    emb2pid = torch.zeros(total_num_tok_embeddings, dtype=torch.int)

    """
    EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element.
    emb2pid_delta will have the delta from the corresponding offset,
    """

    offset_doclens = 0
    for pid, dlength in enumerate(all_item_lens):
        emb2pid[offset_doclens : offset_doclens + dlength] = pid
        offset_doclens += dlength

    if verbose > 1:
        print("len(emb2pid) =", len(emb2pid))

    ivf = emb2pid[orig_ivf]
    unique_pids_per_centroid = []
    ivf_lengths = []

    offset = 0
    for length in tqdm.tqdm(orig_ivf_lengths.tolist()):
        pids = torch.unique(ivf[offset : offset + length])
        unique_pids_per_centroid.append(pids)
        ivf_lengths.append(pids.shape[0])
        offset += length
    ivf = torch.cat(unique_pids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)

    original_ivf_path = os.path.join(index_path, f"{granularity}-ivf.pt")
    optimized_ivf_path = os.path.join(index_path, f"{granularity}-ivf.pid.pt")
    torch.save((ivf, ivf_lengths), optimized_ivf_path)
    if verbose > 1:
        print(f"#> Saved optimized IVF to {optimized_ivf_path}")
        if os.path.exists(original_ivf_path):
            print(f'#> Original IVF at path "{original_ivf_path}" can now be removed')

    return ivf, ivf_lengths
