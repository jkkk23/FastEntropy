import random

import numpy as np
import torch
from torchsparse.nn import functional as F


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torchsparse(kmap_mode="hashmap"):
    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = kmap_mode
    F.conv_config.set_global_conv_config(conv_config)


def select_device(device=None):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
