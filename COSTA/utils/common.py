import random
import os
import numpy as np
import torch
from torch.backends import cudnn
import json

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)

def read_json_dict(path):
    with open(path, "r", encoding='utf-8') as r:
        dic = json.load(r)
    return dic