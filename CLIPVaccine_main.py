import argparse
from utilis import *
from network import *
from noisy_data import *
import logging
import numpy as np
import torch
from pkg_resources import packaging
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchvision.datasets import CIFAR100
import logging
import clip
from torch.cuda.amp import autocast, GradScaler
import math
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='CLIPVaccine')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset', type=str, default='CIFAR100')
parser.add_argument('--network', type=str, default='r34')
parser.add_argument('--optimizer', type=str, default=None)
parser.add_argument('--corruption_type', type=str, default='Flip')
parser.add_argument('--save', action='store_true') 
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--T_init', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--classes', type=int, default=100)
parser.add_argument('--decay_epoch', type=int, nargs='+', default=None)
parser.add_argument('--estimator_interval', type=int, default=3)
parser.add_argument('--data_list_exist', action='store_true')
args =parser.parse_args()


if args.decay_epoch is None:
    decay_epoch1 = args.max_epoch - 10
    reduction_points = [decay_epoch1]
else:
    decay_epoch1, decay_epoch2 = args.decay_epoch[0], args.decay_epoch[1]
    reduction_points = [decay_epoch1, decay_epoch2]

logging.basicConfig(
    filename = f"",       
    level = logging.INFO,
    format = "%(asctime)s - %(message)s"
)

noisy_dataset, label_list, T_real = create_noisy_dataset(args.dataset, args.corruption_type, args.corruption_ratio, args.data_list_exist)
CLIP, preprocess = clip.load("ViT-B/32")
CLIP.cuda().eval()
input_resolution = CLIP.visual.input_resolution
context_length = CLIP.context_length
vocab_size = CLIP.vocab_size
if args.dataset == 'CIFAR100':
    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    loader = DataLoader(cifar100, batch_size=128, shuffle=False)
    text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
elif args.dataset == 'CIFAR10':
    cifar10 = CIFAR10(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    loader = DataLoader(cifar10, batch_size=128, shuffle=False)
    text_descriptions = [f"This is a photo of a {label}" for label in cifar10.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = CLIP.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

class T_init_random(nn.Module):
    def __init__(self, num_classes, init):
        super(T_init_random, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init * torch.ones(num_classes, num_classes)))

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co
        self.identity = torch.eye(num_classes)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig * self.co.detach()
        T = F.normalize(T, p=1, dim=1)
        return T

def T_init_CLIP():
    all_list_probs = []
    all_list_index = []
    index = 0
    for i in range(args.classes):
        class_list = []
        all_list_probs.append(class_list)
    for i in range(args.classes):
        class_list = []
        all_list_index.append(class_list)
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.half()
            with autocast():
                image_features = CLIP.encode_image(images.cuda()).float()
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs, predicted = torch.max(text_probs, 1)
            for i in range(len(probs)):
                all_list_probs[predicted[i]].append(probs[i])
                all_list_index[predicted[i]].append(index)
                index += 1

    combinations = []
    for i in range(args.classes):
        list2 = [x.item() for x in all_list_probs[i]]
        list1 = all_list_index[i]
        combination = combine_lists(list1, list2)
        combinations.append(combination)

    high_index_probs_tuple_list = []
    per = 1

    for i in range(args.classes):
        alist = combinations[i]
        selected_list = select_top_percent(alist, 1, per)
        high_index_probs_tuple_list.append(selected_list)

    T_all = []
    for classes in range(args.classes):
        T_row = []
        selected_nosiy_label = []
        alist = high_index_probs_tuple_list[classes]
        index_list = [x[0] for x in alist]
        for i in index_list:
            selected_nosiy_label.append(label_list[i])
        for i in range(args.classes):
            count = len([x for x in selected_nosiy_label if x == i])
            if len(index_list) != 0:
                T_row.append(count / len(index_list))
            else:
                T_row.append(0)
        T_all.append(T_row)
    return T_all

def get_momentum_T(epoch, end):
    last = end
    if epoch <= last:
        t = epoch
    else:
        t = end
    T_max = 1.
    T_min = 0.
    return T_min + 0.5 * (T_max - T_min) * (
        1 + math.cos(math.pi * (t ** 2) / (last ** 2))  
    )

def get_momentum_KD(epoch):
    if args.optimizer == 'SGD':
        last = 5
        if epoch <= last:
            t = epoch
        else:
            t = last
        T_max = 1
        T_min = 0
        return T_min + 0.5 * (T_max - T_min) * (
            1 + math.cos(math.pi * (t ** 2) / (last ** 2))  
        )
    else:
        last = 150
        if epoch <= last:
            t = epoch
        else:
            t = last
        T_max = 1
        T_min = 0.5
        return T_min + 0.5 * (T_max - T_min) * (
                1 + math.cos(math.pi * (t ** 2) / (last ** 2))  
        )

def T_estimator(epoch, net, T_estimation_original):
    transform_train_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    if args.dataset == 'CIFAR100':
        target_dataset = CIFAR100(os.path.expanduser("~/.cache"), train=True, download=True, transform=transform_train_test)
    elif args.dataset == 'CIFAR10':
        target_dataset = CIFAR10(os.path.expanduser("~/.cache"), train=True, download=True, transform=transform_train_test)

    trainloader_ = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=False)
    all_list_probs = []
    all_list_index = []
    index = 0
    for i in range(args.classes):
        class_list = []
        all_list_probs.append(class_list)
    for i in range(args.classes):
        class_list = []
        all_list_index.append(class_list)

    with torch.no_grad():
        for inputs, targets in trainloader_:
            inputs, targets = inputs.cuda(), targets.cuda()
            net.eval()
            student_features, clean_logits = net(inputs)
            probs, predicted = torch.max(clean_logits, 1)
            for i in range(len(probs)):
                all_list_probs[predicted[i]].append(probs[i])
                all_list_index[predicted[i]].append(index) 
                index += 1

    combinations = []
    for i in range(args.classes):
        list2 = [x.item() for x in all_list_probs[i]]
        list1 = all_list_index[i]
        combination = combine_lists(list1, list2)
        combinations.append(combination)
    high_index_probs_tuple_list = []
    per = 1
    for i in range(args.classes):
        alist = combinations[i]
        selected_list = select_top_percent(alist, 1, per)
        high_index_probs_tuple_list.append(selected_list)

    momentum = get_momentum_T(epoch, decay_epoch1)
    T_all_train = []
    T_all = []
    for classes in range(args.classes):
        T_row = []
        selected_nosiy_label = []
        alist = high_index_probs_tuple_list[classes]
        index_list = [x[0] for x in alist]
        for i in index_list:
            selected_nosiy_label.append(label_list[i])
        for i in range(args.classes):
            count = len([x for x in selected_nosiy_label if x == i])
            if len(index_list) != 0:
                T_row.append(count / len(index_list))
            else:
                T_row.append(0)
        T_train = momentum * T_estimation_original[classes] + (1 - momentum) * torch.tensor(T_row)
        T_train = T_train.tolist()
        T_all.append(T_row)
        T_all_train.append(T_train)
    return T_all_train, T_all

def foward_loss(out, target, T):
    out_softmax = F.softmax(out, dim=1)
    p_T = torch.matmul(out_softmax , T)
    cross_loss = F.nll_loss(torch.log(p_T), target.long())
    return cross_loss, out_softmax, p_T

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def CLIPVaccine():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    ce_loss = torch.nn.CrossEntropyLoss()
    max_test_acc = 0.

    if args.network == "r18":
        net = ResNet18(args.classes).cuda()
    else:
        net = ResNet34(args.classes).cuda()

    if args.T_init == 'CLIP':
        T_estimation_original = torch.tensor(T_init_CLIP()).cuda()
    else:
        trans = T_init_random(num_classes=args.classes, init=4.5)
        T_estimation_original = torch.tensor(trans()).cuda()

    T_train = T_estimation_original

    trainloader = DataLoader(noisy_dataset, batch_size=128, shuffle=True)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    if args.dataset == 'CIFAR100':
        test_dataset = CIFAR100(os.path.expanduser("~/.cache"), train=False, download=True, transform=transform_test)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    elif args.dataset == 'CIFAR10':
        test_dataset = CIFAR10(os.path.expanduser("~/.cache"), train=False, download=True, transform=transform_test)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=1e-3,
        )
        scheduler1 = MultiStepLR(optimizer, milestones=reduction_points, gamma=0.1)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler1 = MultiStepLR(optimizer, milestones=reduction_points, gamma=0.1)

    def train(epoch):
        net.train()
        momentum = get_momentum_KD(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, teacher_features) in enumerate(trainloader):
            inputs, targets, teacher_features = inputs.cuda(), targets.cuda(), teacher_features.cuda()
            optimizer.zero_grad()
            student_features, clean_logits = net(inputs)
            prob = F.softmax(clean_logits, dim=1)
            prob = prob.t()
            distill_loss = 1 - F.cosine_similarity(student_features, teacher_features).mean()
            forward_loss, out_softmax, p_T = foward_loss(clean_logits, targets, T_train.detach())
            loss = (1 - momentum) * forward_loss + momentum * distill_loss
            out_forward = torch.matmul(T_train.t(), prob)
            out_forward = out_forward.t()
            running_loss += loss.item()
            predicted = torch.max(out_forward, 1)[1]
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out_forward.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler1.step()
        print(f"Epoch {epoch}: Loss: {running_loss / len(trainloader)}, Accuracy: {100. * correct / total}")
        logging.info(
            f"TRAIN: Epoch [{epoch}], Average Loss: {running_loss / len(trainloader)}, Accuracy: {100. * correct / total}%"
        )

    def test():
        max = max_test_acc
        net.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _, outputs = net(inputs)
                loss = ce_loss(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss = running_loss / len(testloader)
        test_acc = 100. * correct / total
        if test_acc > max:
            max = test_acc
            file_path = f''
            torch.save(net.state_dict(), file_path)
            print("Model save successfully!")
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        logging.info(
            f"TEST: Average Loss: {running_loss / len(testloader)}, Accuracy: {100. * correct / total}%"
        )
        return test_loss, max

    for epoch in range(0, args.max_epoch):
        train(epoch)
        _, max_test_acc = test()
        if epoch % 3 == 0:
            T_all_train, T_all = T_estimator(epoch, net, T_estimation_original.cpu())
            T_train = torch.tensor(T_all_train).cuda()
            average = error(np.array(T_all), T_real)
            logging.info(
                f"TRAIN: Epoch [{epoch}], T_estimation_error:{average}"
            )
            print(f"The average of delta:{average}")


    print(max_test_acc)
    logging.info(
        f"Max Accuracy: {max_test_acc}%"
    )

if __name__ == '__main__':
    CLIPVaccine()


