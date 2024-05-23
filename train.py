import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bbox_estimator_pointnet import BoxEstimatorPointNet
from models.bbox_estimator_pointnet2 import BoxEstimatorPointNetPlusPlus
from dataset.pointcloud_dataset import PointCloudDataset

# Ensure CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def evaluate_model(model, dataloader):
    """Evaluates the model on the provided dataloader."""
    losses = {
        'total_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    metrics = {
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    model.eval()
    n_batches = len(dataloader)

    with torch.no_grad():
        for data_dicts in tqdm(dataloader, total=n_batches, smoothing=0.9):
            data_dicts = {key: value.cuda() for key, value in data_dicts.items()}
            batch_losses, batch_metrics = model(data_dicts)

            for key in losses:
                if key in batch_losses:
                    losses[key] += batch_losses[key].item()
            for key in metrics:
                if key in batch_metrics:
                    metrics[key] += batch_metrics[key]

    losses = {key: val / n_batches for key, val in losses.items()}
    metrics = {key: val / n_batches for key, val in metrics.items()}

    return losses, metrics


def train_model(model, dataloader, optimizer):
    """Trains the model for one epoch on the provided dataloader."""
    losses = {
        'total_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }
    metrics = {
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    model.train()
    n_batches = len(dataloader)

    for data in tqdm(dataloader, total=n_batches, smoothing=0.9):
        data = {key: value.cuda() for key, value in data.items()}

        optimizer.zero_grad()
        batch_losses, batch_metrics = model(data)
        batch_losses['total_loss'].backward()
        optimizer.step()

        for key in losses:
            if key in batch_losses:
                losses[key] += batch_losses[key].item()
        for key in metrics:
            if key in batch_metrics:
                metrics[key] += batch_metrics[key]

    losses = {key: val / n_batches for key, val in losses.items()}
    metrics = {key: val / n_batches for key, val in metrics.items()}

    return losses, metrics


def save_best_model(model, optimizer, epoch, metrics, best_metrics, save_dir):
    """Saves the model if the current metrics are the best seen so far."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'acc{metrics["iou3d_0.7"]:.3f}-epoch{epoch:03d}.pth')

    if os.path.exists(best_metrics['file']):
        os.remove(best_metrics['file'])
    best_metrics['file'] = save_path

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(state, save_path)


def main():
    # Define model and classes
    classes = ['car', 'truck', 'bus', 'trailer']
    # model = BoxEstimatorPointNetPlusPlus(len(classes)).cuda()
    model = BoxEstimatorPointNet(len(classes)).cuda()

    # Load datasets and dataloaders
    data_dir = "/home/kaan/datas/"
    train_dataset = PointCloudDataset(data_dir, classes, min_points=10, train=True, augment_data=False,
                                      use_mirror=False, use_shift=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = PointCloudDataset(data_dir, classes, min_points=10, train=False, augment_data=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define training parameters
    max_epochs = 200
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_metrics = {'iou3d_0.7': 0.0, 'file': ''}

    # Training loop
    for epoch in range(1, max_epochs + 1):
        train_losses, train_metrics = train_model(model, train_loader, optimizer)
        scheduler.step()

        test_losses, test_metrics = evaluate_model(model, test_loader)

        print(f"Epoch {epoch}/{max_epochs}")
        print(f"Train Losses: {train_losses}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Test Metrics: {test_metrics}")
        print(f"Current Best IoU3D_0.7: {best_metrics['iou3d_0.7']}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")

        if test_metrics['iou3d_0.7'] > best_metrics['iou3d_0.7']:
            best_metrics['iou3d_0.7'] = test_metrics['iou3d_0.7']
            save_best_model(model, optimizer, epoch, test_metrics, best_metrics,
                            save_dir='/home/kaan/projects/bbox_estimator/weights')


if __name__ == "__main__":
    main()
