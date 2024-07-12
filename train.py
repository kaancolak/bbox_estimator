import os
import sys
import time
import argparse
import importlib
import shutil
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.pointnet_bbox_estimator import BBoxEstimator
from models.bbox_estimator_loss import BBoxEstimatorLoss
import dataset.dataset as provider
import datetime

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pointnet_bbox_estimator', help='Model name [default: pointnet_bbox_estimator]')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 201]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=40, help='Decay step for lr decay [default: 60]')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
    parser.add_argument('--ckpt', type=str, default=None, help='Pre-trained model file')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay of Adam [default: 1e-4]')
    parser.add_argument('--name', type=str, default='default', help='tensorboard writer name')
    parser.add_argument('--return_all_loss', default=False, action='store_true', help='only return total loss default')
    parser.add_argument('--dataset_path', type=str, default='/home/kaan/dataset_concat/', help='dataset path')
    parser.add_argument('--min_points', type=int, default=10000, help='Minimum points of model')
    return parser.parse_args()

def setup_directories(log_dir, model_file, train_script):

    now = datetime.datetime.now()
    filename = now.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = log_dir + "/" + filename

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'checkpoint')):
        os.makedirs(os.path.join(log_dir, 'checkpoint'))
    print(f'Copying model file {model_file} and train script {train_script} to log directory {log_dir}')

    shutil.copy(model_file, log_dir)
    shutil.copy(train_script, log_dir)

    return log_dir

def load_datasets(dataset_path, min_points, batch_size):
    train_dataset = provider.PointCloudDataset(dataset_path, ['car', 'truck', 'bus', 'trailer'],
                                               min_points=min_points, train=True, use_rotate=True,
                                               use_mirror=True, use_shift=True)
    test_dataset = provider.PointCloudDataset(dataset_path, ['car', 'truck', 'bus', 'trailer'],
                                              min_points=min_points, train=False, use_rotate=True,
                                              use_mirror=True, use_shift=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_dataloader, test_dataloader

def log_string(log_fout, out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)

def test_one_epoch(model, loader, loss_fn, flags):
    test_metrics = {
        'n_samples': 0,
        'total_loss': 0.0,
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_acc70': 0.0,
        'iou2d_acc70': 0.0,
        'iou2d_acc50': 0.0,
        'mask_loss': 0.0,
        'center_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residuals_normalized_loss': 0.0,
        'size_residuals_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }

    for i, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):

        batch_data, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass, batch_sres, batch_rot_angle, batch_one_hot_vec = data

        batch_data = batch_data.transpose(2, 1).float().cuda()
        batch_label = batch_label.float().cuda()
        batch_center = batch_center.float().cuda()
        batch_hclass = batch_hclass.float().cuda()
        batch_hres = batch_hres.float().cuda()
        batch_sclass = batch_sclass.float().cuda()
        batch_sres = batch_sres.float().cuda()
        batch_rot_angle = batch_rot_angle.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        model.eval()

        with torch.no_grad():
            stage1_center, center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals, center = model( batch_data, batch_one_hot_vec)
            if flags.return_all_loss:
                total_loss, mask_loss, center_loss, heading_class_loss, size_class_loss,
                heading_residuals_normalized_loss, size_residuals_normalized_loss, stage1_center_loss, corners_loss = loss_fn(
                    logits, batch_label, center, batch_center, stage1_center, heading_scores,
                    heading_residuals_normalized, heading_residuals, batch_hclass, batch_hres, size_scores,
                    size_residuals_normalized, size_residuals, batch_sclass, batch_sres)
                test_metrics['mask_loss'] += mask_loss.item()
                test_metrics['center_loss'] += center_loss.item()
                test_metrics['heading_class_loss'] += heading_class_loss.item()
                test_metrics['size_class_loss'] += size_class_loss.item()
                test_metrics['heading_residuals_normalized_loss'] += heading_residuals_normalized_loss.item()
                test_metrics['size_residuals_normalized_loss'] += size_residuals_normalized_loss.item()
                test_metrics['stage1_center_loss'] += stage1_center_loss.item()
                test_metrics['corners_loss'] += corners_loss.item()
            else:
                total_loss = loss_fn(center, batch_center, stage1_center, heading_scores,heading_residuals_normalized,
                                     heading_residuals, batch_hclass, batch_hres, size_scores,
                                     size_residuals_normalized, size_residuals, batch_sclass, batch_sres)

            test_metrics['total_loss'] += total_loss.item()

            iou2ds, iou3ds = provider.compute_box3d_iou(center.cpu().detach().numpy(),
                                                        heading_scores.cpu().detach().numpy(),
                                                        heading_residuals.cpu().detach().numpy(),
                                                        size_scores.cpu().detach().numpy(),
                                                        size_residuals.cpu().detach().numpy(),
                                                        batch_center.cpu().detach().numpy(),
                                                        batch_hclass.cpu().detach().numpy(),
                                                        batch_hres.cpu().detach().numpy(),
                                                        batch_sclass.cpu().detach().numpy(),
                                                        batch_sres.cpu().detach().numpy())
            test_metrics['iou2d'] += np.sum(iou2ds)
            test_metrics['iou3d'] += np.sum(iou3ds)
            test_metrics['iou3d_acc70'] += np.sum(iou3ds >= 0.7)
            test_metrics['iou2d_acc70'] += np.sum(iou2ds >= 0.7)
            test_metrics['iou2d_acc50'] += np.sum(iou2ds >= 0.5)

        test_metrics['n_samples'] += batch_data.shape[0]

    for key in test_metrics:
        if key != 'n_samples':
            test_metrics[key] /= test_metrics['n_samples']

    return test_metrics

def train_one_epoch(model, train_loader, optimizer, loss_fn, flags):
    train_metrics = {
        'n_samples': 0,
        'total_loss': 0.0,
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_acc70': 0.0,
        'iou2d_acc70': 0.0,
        'iou2d_acc50': 0.0,
        'mask_loss': 0.0,
        'center_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residuals_normalized_loss': 0.0,
        'size_residuals_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):

        batch_data, batch_label, batch_center, batch_hclass, batch_hres, batch_sclass, batch_sres, batch_rot_angle, batch_one_hot_vec = data

        batch_data = batch_data.transpose(2, 1).float().cuda()
        batch_label = batch_label.float().cuda()
        batch_center = batch_center.float().cuda()
        batch_hclass = batch_hclass.float().cuda()
        batch_hres = batch_hres.float().cuda()
        batch_sclass = batch_sclass.float().cuda()
        batch_sres = batch_sres.float().cuda()
        batch_rot_angle = batch_rot_angle.float().cuda()
        batch_one_hot_vec = batch_one_hot_vec.float().cuda()

        model.train()

        optimizer.zero_grad()

        (stage1_center, center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores,
         size_residuals_normalized, size_residuals, center) = model(batch_data, batch_one_hot_vec)
        if flags.return_all_loss:
            (total_loss, mask_loss, center_loss, heading_class_loss, size_class_loss, heading_residuals_normalized_loss,
             size_residuals_normalized_loss, stage1_center_loss, corners_loss) = loss_fn(
                center, batch_center, stage1_center, heading_scores, heading_residuals_normalized,
                heading_residuals, batch_hclass, batch_hres, size_scores, size_residuals_normalized, size_residuals,
                batch_sclass, batch_sres)
            train_metrics['mask_loss'] += mask_loss.item()
            train_metrics['center_loss'] += center_loss.item()
            train_metrics['heading_class_loss'] += heading_class_loss.item()
            train_metrics['size_class_loss'] += size_class_loss.item()
            train_metrics['heading_residuals_normalized_loss'] += heading_residuals_normalized_loss.item()
            train_metrics['size_residuals_normalized_loss'] += size_residuals_normalized_loss.item()
            train_metrics['stage1_center_loss'] += stage1_center_loss.item()
            train_metrics['corners_loss'] += corners_loss.item()
        else:
            total_loss = loss_fn(center, batch_center, stage1_center, heading_scores,
                                 heading_residuals_normalized, heading_residuals, batch_hclass, batch_hres, size_scores,
                                 size_residuals_normalized, size_residuals, batch_sclass, batch_sres)

        total_loss.backward()
        optimizer.step()

        train_metrics['total_loss'] += total_loss.item()

        iou2ds, iou3ds = provider.compute_box3d_iou(center.cpu().detach().numpy(),
                                                    heading_scores.cpu().detach().numpy(),
                                                    heading_residuals.cpu().detach().numpy(),
                                                    size_scores.cpu().detach().numpy(),
                                                    size_residuals.cpu().detach().numpy(),
                                                    batch_center.cpu().detach().numpy(),
                                                    batch_hclass.cpu().detach().numpy(),
                                                    batch_hres.cpu().detach().numpy(),
                                                    batch_sclass.cpu().detach().numpy(),
                                                    batch_sres.cpu().detach().numpy())
        train_metrics['iou2d'] += np.sum(iou2ds)
        train_metrics['iou3d'] += np.sum(iou3ds)
        train_metrics['iou3d_acc70'] += np.sum(iou3ds >= 0.7)
        train_metrics['iou2d_acc70'] += np.sum(iou2ds >= 0.7)
        train_metrics['iou2d_acc50'] += np.sum(iou2ds >= 0.5)

        train_metrics['n_samples'] += batch_data.shape[0]

    for key in train_metrics:
        if key != 'n_samples':
            train_metrics[key] /= train_metrics['n_samples']

    return train_metrics

def adjust_learning_rate(optimizer, epoch, base_lr, decay_step, decay_rate):
    lr = base_lr * (decay_rate ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    args = parse_arguments()
    set_random_seeds()
    log_dir = setup_directories(args.log_dir, f'models/{args.model}.py', 'train.py')

    writer = SummaryWriter(log_dir=os.path.join(log_dir, args.name))
    log_fout = open(os.path.join(log_dir, 'log_train.txt'), 'w')
    log_string(log_fout, str(args))

    model = BBoxEstimator(n_classes=3 if args.no_intensity else 4, n_channel=3).cuda()

    loss_fn = BBoxEstimatorLoss(return_all=args.return_all_loss).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay) if args.optimizer == 'adam' else torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt))

    train_loader, test_loader = load_datasets(args.dataset_path, args.min_points, args.batch_size)

    best_accuracy = 0.0

    for epoch in range(args.max_epoch):
        log_string(log_fout, f'**** EPOCH {epoch:03d} ****')
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.decay_step, args.decay_rate)
        log_string(log_fout, f'Learning rate: {lr:.6f}')

        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, args)
        test_metrics = test_one_epoch(model, test_loader, loss_fn, args)

        log_string(log_fout, f"Training metrics: {train_metrics}")
        log_string(log_fout, f"Testing metrics: {test_metrics}")

        writer.add_scalar('Loss/train', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/test', test_metrics['total_loss'], epoch)
        writer.add_scalar('IoU3D/test', test_metrics['iou3d'], epoch)
        writer.add_scalar('iou3d_acc70/test', test_metrics['iou3d_acc70'], epoch)
        writer.add_scalar('IoU2D/test', test_metrics['iou2d'], epoch)
        writer.add_scalar('iou2d_acc70/test', test_metrics['iou2d_acc70'], epoch)
        writer.add_scalar('IoU2D_acc50/test', test_metrics['iou2d_acc50'], epoch)

        # Save the model if test accuracy is higher than previous best
        current_accuracy = test_metrics['iou3d_acc70']
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            accuracy_str = "{:.2f}".format(best_accuracy)
            filename = f"model_accuracy_{accuracy_str}.pth"
            savepath = os.path.join(log_dir, 'checkpoint', filename)
            log_string(log_fout, f'Saving model at {savepath}')
            torch.save(model.state_dict(), savepath)

    log_fout.close()
    writer.close()

if __name__ == '__main__':
    main()
