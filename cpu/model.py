import torch
import torch.nn as nn
from transformers import ViTModel

class DinoV2Model(nn.Module):
    def __init__(self, num_classes=10):
        super(DinoV2Model, self).__init__()
        self.vit = ViTModel.from_pretrained('facebook/dino-vits8')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, x):
        outputs = self.vit(x)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits
