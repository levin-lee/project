from data import load_data
from mydataset import MyDataset
from resnet import ResNet, weight_init
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
import cv2
from PIL import Image
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, criterion, optimizer, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader)
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            # labels -= 1  # Subtract 1 from all class labels
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1} train loss: {train_loss/(1 + i)}")
            
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # labels -= 1  # Subtract 1 from all class labels
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                # print(outputs)
                # print(labels)
                # print('++++++++++++++++++++++++')
                correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(val_loader)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {correct / len(val_loader.dataset):.4f}')
        

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5,0.5,0.5],
                                    std=[0.5,0.5,0.5]
                                ),
                                transforms.Resize((224, 224))
                               ])
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

testing_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)
# 获取前1000个样本的索引
indices = range(100)
# 使用Subset进行切片
training_data = torch.utils.data.Subset(training_data, indices)
testing_data = torch.utils.data.Subset(testing_data, indices)

batch_size=8
train_data = DataLoader(dataset=training_data,batch_size=batch_size,shuffle=True,drop_last=True)
test_data = DataLoader(dataset=testing_data,batch_size=batch_size,shuffle=True,drop_last=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# model.apply(weight_init)
train(model, criterion, optimizer, train_data, test_data, epochs=100)

