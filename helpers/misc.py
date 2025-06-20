import datetime
from typing import MutableMapping
import dataclasses

import numpy as np
import torch


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def flatten_dict(dictionary, parent_key='', separator='.'):
    """
    Flattens a nested dictionary.

    Parameters
    ----------
    dictionary: dict
        The dictionary to be flattened.
    parent_key: str
        The parent key to be prepended to the key names. Usually this should be an empty string when
        the function is called by user. This argument is more intended for recursion.
    separator: str
        The separator delimiting the key names of different nesting levels.

    Returns
    -------
    Dict
        The flattened dictionary.
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def serialize_dict(d):
    for k, v in d.items():
        if dataclasses.is_dataclass(v.__class__):
            d[k] = serialize_dict(v.__dict__)
        elif isinstance(v, dict):
            d[k] = serialize_dict(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach().cpu().numpy().tolist()
    return d