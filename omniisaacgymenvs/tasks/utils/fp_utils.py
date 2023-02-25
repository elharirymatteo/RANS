import torch

def quantize_tensor_values(tensor, n_values):
    """
    Quantizes the values of a tensor into N discrete values in the range [-1,1].

    Args:
    - tensor: a PyTorch tensor of shape (batch_size, num_features)
    - n_values: an integer indicating the number of discrete values to use

    Returns:
    - a new tensor of the same shape as the input tensor, with each value quantized to a discrete value in the range [-1,1]
    """
    assert n_values >= 2, "n_values must be greater than or equal to 2"

    # Calculate the bin width based on the number of discrete values
    bin_width = 2.0 / (n_values - 1)

    # Compute the bins for quantization
    bins = torch.linspace(-1, 1, n_values)

    # Compute the index of the bin for each element of the input tensor
    bin_idx = ((tensor - bins[0]) / bin_width).long().clamp(0, n_values - 2)

    # Compute the quantized values for each element of the input tensor
    quantized_tensor = -1 + bin_idx * bin_width

    return quantized_tensor
