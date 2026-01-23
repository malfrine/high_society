import numpy as np
import torch

def cat_dict_array(d: dict[str, np.array]) -> np.array:
    return np.concatenate(
        [d[k] for k in sorted(d.keys())],
        axis=0
    )

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
