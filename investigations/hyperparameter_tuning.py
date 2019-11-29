import numpy as np
import random
import utils

import torch
from torch.utils.data import TensorDataset

from custom_dataset import CustomImageTensorDataset
from data_augmentation import get_transform


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.enabled = False

    return True


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")

# Train and test datasets
dataset_train = CustomImageTensorDataset('custom_dataset', get_transform(train=True))
dataset_test = CustomImageTensorDataset('custom_dataset', get_transform(train=False))

# split the dataset in train and test set (0.8 - 0.2)
torch.manual_seed(1)
indices = torch.randperm(len(dataset_train)).tolist()
dataset_train = torch.utils.data.Subset(dataset_train, indices[:-int(
    0.2 * len(dataset_train))])  # to be changed to respect the proportions
dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(
    0.2 * len(dataset_train)):])  # to be changed to respect the proportions

# split the train dataset into training and validation dataset
index = torch.randperm(len(dataset_train)).tolist()
dataset_tr = torch.utils.data.Subset(dataset_train, index[:-int(
    0.2 * len(dataset_train))])  # to be changed to respect the proportions
dataset_val = torch.utils.data.Subset(dataset_train, index[-int(
    0.2 * len(dataset_train)):])  # to be changed to respect the proportions

data_loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=2, shuffle=True, num_workers=4,
                                             collate_fn=utils.collate_fn, pin_memory=True)

data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,
                                              collate_fn=utils.collate_fn, pin_memory=True)


import adabound


# opt = [torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005),
#        torch.optim.Adam(params, lr=0.005, weight_decay=0.0005), adabound.AdaBound(params, lr=5e-3)]

# mom_vals = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
wd = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]

# wd = 0.0005

# lrs = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005] #Adam

lrs = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075] #SGD

lrs_ab = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005]

# for boo in [False, True]:
for lr in lrs:
    print("lr tested: ", lr)
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
    # lr = 5e-3
    # momentum = 0.9

    # Device choice
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    import time
    from datetime import timedelta

    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=mom, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.0005, weight_decay=0.0005, amsgrad=boo)

    # optimizer = adabound.AdaBound(params, lr= 0.0005, final_lr=lr)
    optimizer = adabound.AdaBound(params, lr=lr, final_lr=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    start_time = time.monotonic()

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_tr, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))

    print("END OF CONFIG")
    print("")
