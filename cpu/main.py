import torch
from model import DinoV2Model
from test_loader import test_loader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

# load the pre-trained model
model = DinoV2Model(num_classes=10)
model.load_state_dict(torch.load('dino_v2_trained_model.pth'))
model.eval()  # set model to evaluation mode

# cpu
device = torch.device('cpu')
model.to(device)

# loss function and test variables
criterion = torch.nn.CrossEntropyLoss()
correct = 0
total = 0
running_loss = 0.0
all_preds = []
all_labels = []

progress_bar = tqdm(test_loader, desc="Testing", unit="batch")

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Get model predictions
        outputs = model(images)
        loss = criterion(outputs, labels)  # Compute the loss
        running_loss += loss.item()

        # Get predicted class
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store predictions and true labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar with the current loss and accuracy
        progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

# accuracy and loss
test_loss = running_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
