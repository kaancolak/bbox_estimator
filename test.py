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

    weights = torch.load('/home/kaan/projects/bbox_estimator/weights/v1_min10_acc0.610-epoch049.pth')
    model.load_state_dict(weights['model_state_dict'], strict=False)

    # Load datasets and dataloaders
    data_dir = "/home/kaan/custom_dataset/"

    test_dataset = PointCloudDataset(data_dir, classes, min_points=10, train=True, augment_data=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test_losses, test_metrics = evaluate_model(model, test_loader)

    print(test_losses)
    print(test_metrics)

if __name__ == "__main__":
    main()
