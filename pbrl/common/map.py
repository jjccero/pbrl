import torch


def treemap(f, d: dict):
    return {k: treemap(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}


def listmap(f, l: list):
    return [listmap(f, e) if isinstance(e, list) else f(e) for e in l]


def map_cpu(e):
    if isinstance(e, torch.Tensor):
        return e.cpu()
    return e
