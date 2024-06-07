import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

from models.bbox_estimator_pointnet import BoxEstimatorPointNet
from models.bbox_estimator_pointnet2 import BoxEstimatorPointNetPlusPlus
from models.bbox_estimator_pointnet2_cuda import BoxEstimatorPointNetPlusPlusCuda
from dataset.pointcloud_dataset import PointCloudDataset

# Ensure CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_metric(name, value):
    """Logs a metric to Vertex AI."""
    logger.info(f'Metric: {{name: "{name}", value: {value}}}')


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


class Config:
    def __init__(self, classes, data_dir, data_dir_test, batch_size, lr, betas, eps, weight_decay, step_size, gamma,
                 max_epochs, save_dir):
        self.classes = classes
        self.data_dir = data_dir
        self.data_dir_test = data_dir_test
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.save_dir = save_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for bbox estimator")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--step_size', type=int, default=20, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma for LR scheduler')
    parser.add_argument('--max_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='/gcs/pc-shape-estimation/weights/',
                        help='Directory to save models')
    parser.add_argument('--data_dir', type=str, default='/gcs/pc-shape-estimation/dataset_concat/',
                        help='Directory to save models')
    args = parser.parse_args()
    return args


def main(config):
    # Define model and classes
    # model = BoxEstimatorPointNetPlusPlus(len(config.classes)).cuda()
    model = BoxEstimatorPointNet(len(config.classes)).cuda()
    # model = BoxEstimatorPointNetPlusPlusCuda(len(config.classes)).cuda()

    # weights = torch.load('/home/kaan/projects/bbox_estimator/weights/v1_min10_acc0.610-epoch049.pth')
    # model.load_state_dict(weights['model_state_dict'], strict=False)

    # Load datasets and dataloaders
    train_dataset = PointCloudDataset(config.data_dir, config.classes, min_points=10, train=True, augment_data=False,
                                      use_mirror=False, use_shift=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = PointCloudDataset(config.data_dir_test, config.classes, min_points=10, train=False,
                                     augment_data=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Define training parameters
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    best_metrics = {'iou3d_0.7': 0.0, 'file': ''}

    # Training loop
    for epoch in range(1, config.max_epochs + 1):
        train_losses, train_metrics = train_model(model, train_loader, optimizer)
        scheduler.step()

        test_losses, test_metrics = evaluate_model(model, test_loader)

        print(f"Epoch {epoch}/{config.max_epochs}")
        print(f"Train Losses: {train_losses}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Test Metrics: {test_metrics}")
        print(f"Current Best IoU3D_0.7: {best_metrics['iou3d_0.7']}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")

        # Log the metric for Vertex AI
        log_metric('accuracy', test_metrics['iou3d_0.7'])

        if test_metrics['iou3d_0.7'] > best_metrics['iou3d_0.7']:
            best_metrics['iou3d_0.7'] = test_metrics['iou3d_0.7']
            save_best_model(model, optimizer, epoch, test_metrics, best_metrics, save_dir=config.save_dir)


if __name__ == "__main__":
    args = parse_args()

    # Define hyperparameters
    config = Config(
        classes=['car', 'truck', 'bus', 'trailer'],
        data_dir=args.data_dir,
        data_dir_test=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        max_epochs=args.max_epochs,
        save_dir=args.save_dir
    )

    main(config)