def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Same contract as kaolin.utils.testing.check_tensor: None in shape matches any size."""
    ok = True
    if shape is not None:
        ok = len(tensor.shape) == len(shape) and all(s is None or s == t for s, t in zip(shape, tensor.shape))
    if ok and dtype is not None:
        ok = tensor.dtype == dtype
    if ok and device is not None:
        ok = tensor.device == device
    if not ok and throw:
        raise ValueError(f"tensor check failed: shape {tuple(tensor.shape)} vs {shape}")
    return ok
