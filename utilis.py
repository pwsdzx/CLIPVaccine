import random
import numpy as np
import torch

def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = (device == 'cuda')
    torch.backends.cudnn.benchmark = (device == 'cuda')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def combine_lists(list1, list2):
    result = list(zip(list1, list2))
    return result

def select_top_percent(lst, index, per):
    sorted_lst = sorted(lst, key=lambda x: x[index], reverse=True)
    num_to_select = int(len(sorted_lst) * per)
    return sorted_lst[:num_to_select]
