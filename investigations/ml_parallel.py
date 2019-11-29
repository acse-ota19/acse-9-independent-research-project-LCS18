#!/home/project8/hpc_fluids/pytorch/bin/python
import numpy as np
import random
import utils

from torch.utils.data import TensorDataset

from custom_dataset import CustomImageTensorDataset
from data_augmentation import get_transform

from torch.nn.parallel import DistributedDataParallel as DDP

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example pytorch.distributed initialisation
import torch.distributed as dist
import os
import argparse
import torch

# Read the environment variables
world_size = os.getenv("SLURM_STEP_NUM_TASKS")
print(world_size)

rank = os.getenv("SLURM_PROCID")
print(rank)

gpu_id = int(os.getenv("SLURM_LOCALID"))
print(gpu_id)

# Read in the rank 0 hostname
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, help="master host")
args = parser.parse_args()

# Initialise the parallel process group and bind to the GPU.
dist.init_process_group(backend="nccl", world_size=world_size, rank=rank, init_method="tcp://%s:12345" % args.m)
device = torch.device("cuda:%i" % gpu_id)
torch.cuda.set_device(device)

size = dist.get_world_size()
rank = dist.get_rank()
print("I am rank %i of %i" % (rank, size))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = False

    return True


# Train and test datasets
dataset_train = CustomImageTensorDataset('custom_dataset', get_transform(train=True))
dataset_test = CustomImageTensorDataset('custom_dataset', get_transform(train=False))

# print(-int(0.2 * len(dataset_train)))

# split the dataset in train and test set (0.8 - 0.2)
torch.manual_seed(1)
# indices = torch.randperm(len(dataset_train)).tolist()

indices = [i for i in range(len(dataset_train))]

dataset_train = torch.utils.data.Subset(dataset_train, indices[:-int(
    0.2 * len(dataset_train))])  # to be changed to respect the proportions
dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(
    0.2 * len(dataset_train)):])  # to be changed to respect the proportions

# Create a sampler to split the input data
parallel_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=size, rank=rank)

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=False, num_workers=8,
                                                collate_fn=utils.collate_fn, pin_memory=True,
                                                sampler=parallel_sampler)  # adding parallel_sampler to do distributed parallelism

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8,
                                               collate_fn=utils.collate_fn, pin_memory=True)

# Finetuning from a pretrained model (Faster R-CNN)
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Doing the training and check the results
from engine import train_one_epoch, evaluate

# Hyperparameters
seed = 42
lr = 0.005
momentum = 0.9
weight_decay = 0.0005

# our dataset has two classes only - background and person
num_classes = 2

# move model to the right device
model.to(device)
model = DDP(model, device_ids=[gpu_id])

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

import time
from datetime import timedelta

start_time = time.monotonic()

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

# # Saving the model
# model_save_name = 'faster-r-cnn-resnet50-fpn-finetuning.pt'
# path = model_save_name
# torch.save(model.state_dict(), path)
