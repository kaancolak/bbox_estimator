from models.bbox_estimator import PointNet2BoundingBox
from dataset.pointcloud_dataset import PointCloudDataset
from loss.iou_loss import compute_ciou_loss
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

classes = ['car', 'truck', 'bus', 'trailer']
# # classes = ['car', 'pedestrian', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle']
model = PointNet2BoundingBox(len(classes))

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


datadir = "/media/kaan/Extreme SSD/nuscenes/"
dataset = PointCloudDataset(datadir, classes)

print(len(dataset))
print(dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for i, (points, labels) in enumerate(dataloader):
#     print(points.shape, labels.shape)
#     break


def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (point_cloud, one_hot_vec, ground_truth_boxes) in enumerate(dataloader):
            optimizer.zero_grad()
            # Forward pass
            predicted_boxes = model(point_cloud, one_hot_vec)
            # Compute CIoU loss
            loss = criterion(predicted_boxes, ground_truth_boxes)
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")



# Define your CIoU loss function
criterion = compute_ciou_loss

# Train your model
train(model, dataloader, optimizer, criterion, epochs=10)