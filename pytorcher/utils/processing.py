import torch

def normalize(tensor: torch.Tensor, return_min_max=False) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1].

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    tensor_norm = (tensor - min_val) / (max_val - min_val)
    if return_min_max:
        return tensor_norm, min_val, max_val
    else:
        return tensor_norm

def normalize_batch(batch_tensor: torch.Tensor, batch_dim=0, return_min_max=False) -> torch.Tensor:
    """
    Normalize a batch of tensors to the range [0, 1] for each tensor in the batch.

    Parameters
    ----------
    batch_tensor : torch.Tensor
        Input batch tensor of shape (B, C, H, W) or (B, H, W).

    Returns
    -------
    torch.Tensor
        Normalized batch tensor.
    """
    if return_min_max:
        min_vals = torch.empty(batch_tensor.size(batch_dim), device=batch_tensor.device)
        max_vals = torch.empty(batch_tensor.size(batch_dim), device=batch_tensor.device)
    batch_size = batch_tensor.size(batch_dim)
    normalized_batch = torch.empty_like(batch_tensor)
    for i in range(batch_size):
        tensor = batch_tensor.select(batch_dim, i)
        if return_min_max:
            tensor, min_val, max_val = normalize(tensor, return_min_max=True)
            min_vals[i] = min_val
            max_vals[i] = max_val
        else:
            tensor = normalize(tensor)
        normalized_batch.select(batch_dim, i).copy_(tensor)
    if return_min_max:
        return normalized_batch, min_vals, max_vals
    else:
        return normalized_batch

def rescale_batch(batch_tensor: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor, batch_dim=0) -> torch.Tensor:
    """
    Rescale a batch of normalized tensors from the range [0, 1] back to their original ranges.

    Parameters
    ----------
    batch_tensor : torch.Tensor
        Input batch tensor of shape (B, C, H, W) or (B, H, W).
    min_vals : torch.Tensor
        Tensor of minimum values for each tensor in the batch.
    max_vals : torch.Tensor
        Tensor of maximum values for each tensor in the batch.

    Returns
    -------
    torch.Tensor
        Rescaled batch tensor.
    """
    batch_size = batch_tensor.size(batch_dim)
    rescaled_batch = torch.empty_like(batch_tensor)
    for i in range(batch_size):
        tensor = batch_tensor.select(batch_dim, i)
        min_val = min_vals[i]
        max_val = max_vals[i]
        tensor_rescaled = tensor * (max_val - min_val) + min_val
        rescaled_batch.select(batch_dim, i).copy_(tensor_rescaled)
    return rescaled_batch

if __name__ == "__main__":
    # Test the normalize function
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalized_tensor = normalize(tensor)
    print("Original Tensor:\n", tensor)
    print("Normalized Tensor:\n", normalized_tensor)

    # Test the normalize_batch function
    batch_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]])
    normalized_batch = normalize_batch(batch_tensor)
    print("Original Batch Tensor:\n", batch_tensor)
    print("Normalized Batch Tensor:\n", normalized_batch)