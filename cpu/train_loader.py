import os
import torch
from torchvision import datasets, transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINOv2 uses 224x224 images
    transforms.ToTensor(),
])

# Load the training dataset
train_data_dir = 'D:/dinosaur/EuroSAT_RGB/train'
train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)

# Create the DataLoader for training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Expose the train_loader
if __name__ == "__main__":
    print(f"Train samples: {len(train_loader.dataset)}")
