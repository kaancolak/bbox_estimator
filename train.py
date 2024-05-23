from models.bbox_estimator import FrustumPointNetv1
# from models.bbox_estimator_v2 import FrustumPointNetv2
from dataset.pointcloud_dataset import PointCloudDataset
from loss.iou_loss import compute_ciou_loss
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from timeit import default_timer as timer

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def test_one_epoch(model, loader):
    test_losses = {
        'total_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }

    test_metrics = {
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


def weights_init_he(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


classes = ['car', 'truck', 'bus', 'trailer']
# classes = ['car', 'truck', 'bus']
# classes = ['car']
# # classes = ['car', 'pedestrian', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle']
model = FrustumPointNetv1(len(classes))
model.to('cuda')
# model.apply(weights_init_he)

datadir = "/home/kaan/datas/"
dataset_train = PointCloudDataset(datadir, classes, min_points=10, train=True, augment_data=False, use_mirror=False,
                                  use_shift=False)
train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)

dataset_test = PointCloudDataset(datadir, classes, min_points=10, train=False, augment_data=False)
test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

max_epochs = 200

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001,
    betas=(0.9, 0.999), eps=1e-08,
    weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

best_iou3d_70 = 0.0
best_epoch = 1
best_file = ''

for epoch in range(max_epochs):
    train_losses = {
        'total_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    train_metrics = {
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }
    n_batches = 0

    for batch_idx, (data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
        # start = timer()
        n_batches += 1
        data_dicts_var = {key: value.cuda() for key, value in data.items()}

        model = model.train()
        optimizer.zero_grad()
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

    scheduler.step()
    for key in train_losses.keys():
        train_losses[key] /= n_batches
    for key in train_metrics.keys():
        train_metrics[key] /= n_batches

    test_losses, test_metrics = test_one_epoch(model, test_dataloader)

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("Best_iou3d_70")
    print(best_iou3d_70)

    print(scheduler.get_last_lr())

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

    print(f"Epoch {epoch} Train Loss: {train_losses} \nTrain Metrics: {train_metrics}")
