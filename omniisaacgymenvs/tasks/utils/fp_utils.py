import torch


def quantize_tensor_values(tensor, n_values):
    """
    Quantizes the values of a tensor into N*2 +1  discrete values in the range [-1,1] using PyTorch's quantization functions.

    Args:
    - tensor: a PyTorch tensor of shape (batch_size, num_features)
    - n_values: an integer indicating the number of discrete values to use

    Returns:
    - a new tensor of the same shape as the input tensor, with each value quantized to a discrete value in the range [-1,1]
    """
    assert n_values >= 1, "n_values must be greater than or equal to 1"
    assert tensor.min() >= -1 and tensor.max() <= 1, "tensor values must be in the range [-1,1]"
    scale = 1.0 /  n_values

    quantized_tensor = torch.quantize_per_tensor(tensor, scale=scale, zero_point=0, 
                                                 dtype=torch.qint8)
    quantized_tensor = quantized_tensor.dequantize()

    return quantized_tensor
