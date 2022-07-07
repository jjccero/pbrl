import torch


def automap(f, x):
    if isinstance(x, tuple):
        return tuple(automap(f, e) for e in x)
    elif isinstance(x, list):
        return list(automap(f, e) for e in x)
    elif isinstance(x, dict):
        return {k: automap(f, v) for k, v in x.items()}
    else:
        return f(x)


def map_cpu(e):
    if isinstance(e, torch.Tensor):
        return e.cpu()
    return e
