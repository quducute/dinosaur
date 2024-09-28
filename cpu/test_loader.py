import os
import torch
from torchvision import datasets, transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # DINOv2 uses 224x224 images
    transforms.ToTensor(),
])

# Load the testing dataset
test_data_dir = 'D:/dinosaur/EuroSAT_RGB/test'
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

# Create the DataLoader for testing
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Expose the test_loader
if __name__ == "__main__":
    print(f"Test samples: {len(test_loader.dataset)}")
