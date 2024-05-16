from models.bbox_estimator import FrustumPointNetv1
from dataset.pointcloud_dataset import PointCloudDataset
from loss.iou_loss import compute_ciou_loss
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

import os


def test_one_epoch(model, loader):
    test_losses = {
        'total_loss': 0.0,
        'cls_loss': 0.0,  # fconvnet
        'mask_loss': 0.0,  # fpointnet
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }

    test_metrics = {
        'seg_acc': 0.0,  # fpointnet
        'cls_acc': 0.0,  # fconvnet
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    for i, data_dicts in tqdm(enumerate(loader), \
                              total=len(loader), smoothing=0.9):
        n_batches += 1

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
        model = model.eval()

        with torch.no_grad():
            losses, metrics = model(data_dicts_var)

        for key in test_losses.keys():
            if key in losses.keys():
                test_losses[key] += losses[key].detach().item()
        for key in test_metrics.keys():
            if key in metrics.keys():
                test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= n_batches
    for key in test_metrics.keys():
        test_metrics[key] /= n_batches

    return test_losses, test_metrics


# classes = ['car', 'truck', 'bus', 'trailer']
classes = ['car']
# # classes = ['car', 'pedestrian', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle']
model = FrustumPointNetv1(len(classes))
model.to('cuda')
# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

datadir = "/home/kaan/datas/"
dataset = PointCloudDataset(datadir, classes)

dataset_size = len(dataset)
train_size = int(0.70 * dataset_size)
val_size = int(0.10 * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                        [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataload = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

max_epochs = 150

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001,
    betas=(0.9, 0.999), eps=1e-08,
    weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.7)

from timeit import default_timer as timer

best_iou3d_70 = 0.0
best_epoch = 1
best_file = ''

for epoch in range(max_epochs):
    train_losses = {
        'total_loss': 0.0,
        'cls_loss': 0.0,  # fconvnet
        'mask_loss': 0.0,  # fpointnet
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    train_metrics = {
        'cls_acc': 0.0,  # fconvnet
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }
    n_batches = 0

    for batch_idx, (data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
        # start = timer()
        n_batches += 1
        data_dicts_var = {key: value.cuda() for key, value in data.items()}
        optimizer.zero_grad()
        model = model.train()

        losses, metrics = model(data_dicts_var)

        total_loss = losses['total_loss']
        # total_loss = total_loss.mean()
        total_loss.backward()

        optimizer.step()

        for key in train_losses.keys():
            if key in losses.keys():
                train_losses[key] += losses[key].detach().item()
        for key in train_metrics.keys():
            if key in metrics.keys():
                train_metrics[key] += metrics[key]
        # end = timer()
        # print("Total elapsed time:")
        # print(end - start)
    for key in train_losses.keys():
        train_losses[key] /= n_batches
    for key in train_metrics.keys():
        train_metrics[key] /= n_batches

    test_losses, test_metrics = test_one_epoch(model, test_dataloader)

    print("test_metrics['iou3d_0.7']")
    print(test_metrics['iou3d_0.7'])
    print("best_iou3d_70")
    print(best_iou3d_70)

    if test_metrics['iou3d_0.7'] >= best_iou3d_70:
        best_iou3d_70 = test_metrics['iou3d_0.7']
        best_epoch = epoch + 1

        directory = '/home/kaan/projects/bbox_estimator/weights'
        savepath = directory + '/acc%.3f-epoch%03d.pth' % \
                   (test_metrics['iou3d_0.7'], epoch)

        if os.path.exists(best_file):
            os.remove(best_file)  # update to newest best epoch
        best_file = savepath
        state = {
            'epoch': epoch + 1,
            'train_iou3d_0.7': train_metrics['iou3d_0.7'],
            'test_iou3d_0.7': test_metrics['iou3d_0.7'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

    print(f"Epoch {epoch} Train Loss: {train_losses} Train Metrics: {train_metrics}")



