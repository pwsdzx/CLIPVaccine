import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, CIFAR10
import os
from torchvision import transforms
import pickle
import clip

def create_data_list(target_dataset):
    CLIP, preprocess = clip.load("ViT-B/32")
    CLIP.cuda().eval()

    if target_dataset == 'CIFAR100':
        cifar100_train = CIFAR100(os.path.expanduser("~/.cache"), train=True, download=True) 
        data_list = []
        for i in range(len(cifar100_train)):
            data = cifar100_train[i][0]
            input = preprocess(cifar100_train[i][0]).half().unsqueeze(0).cuda()
            teacher_feature = CLIP.encode_image(input).detach()
            tuple = (data, teacher_feature)
            data_list.append(tuple)
        with open('',
                  'wb') as f:
            pickle.dump(data_list, f)
        return data_list
    elif target_dataset == 'CIFAR10':
        cifar10_train = CIFAR10(os.path.expanduser("~/.cache"), train=True, download=True)  
        data_list = []
        for i in range(len(cifar10_train)):
            data = cifar10_train[i][0]
            input = preprocess(cifar10_train[i][0]).half().unsqueeze(0).cuda()
            teacher_feature = CLIP.encode_image(input).detach()
            tuple = (data, teacher_feature)
            data_list.append(tuple)
        with open('',
                  'wb') as f:
            pickle.dump(data_list, f)
        return data_list

def flip_corruption(corruption_ratio, num_classes):
    P = np.eye(num_classes)
    n = corruption_ratio
    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n 
        for i in range(1, num_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[num_classes-1, num_classes-1], P[num_classes-1, 0] = 1. - n, n
    return P

def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix

class noisy_dataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list
        self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  
                transforms.RandomHorizontalFlip(),     
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  
             ])
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample, teacher_feature = self.data_list[idx]
        student_sample = self.transform_train(sample)
        label = int(self.label_list[idx])
        return student_sample, label, teacher_feature.squeeze(0)

def create_noisy_dataset(target_dataset, noise_type, noise_ratio, data_list_exist):
    if target_dataset == 'CIFAR100':
        dataset = CIFAR100(os.path.expanduser("~/.cache"), train=True, download=True)
    elif target_dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.expanduser("~/.cache"), train=True, download=True)

    if noise_type == 'Flip':
        T_real = flip_corruption(noise_ratio, len(dataset.classes))
    elif noise_type == 'Uniform':
        T_real = uniform_corruption(noise_ratio, len(dataset.classes))

    label_clean = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_clean.append(label)
    classes = np.arange(0, len(dataset.classes))
    label_noisy = torch.ones(len(dataset))
    for i in range(len(label_clean)):
        label_noisy[i] = np.random.choice(classes, p=T_real[label_clean[i]])
    label_list = []
    for i in range(len(dataset)):
        label_list.append(label_noisy[i])

    if data_list_exist == True:
        if target_dataset == 'CIFAR100':
            with open('',   
                  'rb') as f:
                data_list = pickle.load(f)
        elif target_dataset == 'CIFAR10':
            with open('',   
                  'rb') as f:
                data_list = pickle.load(f)
    else:
        print('Obtaining the sample representations from CLIP.')
        data_list = create_data_list(target_dataset)
        print('Finish.')
    return noisy_dataset(data_list, label_list), label_list, T_real
