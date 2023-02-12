import torch


def auto_map(f, x, **kwargs):
    if isinstance(x, tuple):
        return tuple(auto_map(f, e, **kwargs) for e in x)
    elif isinstance(x, dict):
        return {k: auto_map(f, v, **kwargs) for k, v in x.items()}
    else:
        return f(x, **kwargs)


def merge_map(f, x, **kwargs):
    # x must be iterable
    first_item = x[0]
    if isinstance(first_item, tuple):
        return tuple(merge_map(f, e, **kwargs) for e in zip(*x))
    elif isinstance(first_item, dict):
        return {k: merge_map(f, tuple(e[k] for e in x), **kwargs) for k in first_item}
    else:
        return f(x, **kwargs)


def map_cpu(e):
    if isinstance(e, torch.Tensor):
        return e.cpu()
    return e
