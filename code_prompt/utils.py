import random

import torch
import numpy as np


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.

    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
