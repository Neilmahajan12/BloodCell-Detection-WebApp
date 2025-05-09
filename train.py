# train.py

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

from dataset import BCCDDataset
from transforms import get_transform

# Setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4  # 3 classes (RBC, WBC, Platelets) + background

# Paths
images_dir = 'data/images/train'
annotations_dir = 'data/annotations/train'

# Dataset and DataLoader
dataset = BCCDDataset(images_dir, annotations_dir, transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pretrained model and modify classifier head
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer and LR Scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the fine-tuned model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/fine_tuned_bccd.pth")
print("Model saved to models/fine_tuned_bccd.pth")
