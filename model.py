import torch.nn as nn
import torchvision.models as models

# Load a pre-trained MobileNet and customize it for CIFAR-10
def load_mobilenet(num_classes=10):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
