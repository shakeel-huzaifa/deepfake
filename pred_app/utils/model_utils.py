import torchvision.models as models
import torch.nn as nn
import torch
import os
import timm
from django.conf import settings


# Define functions to load your trained models (replace with your specific code)
def load_resnet50_model():
    # 1. Load the pre-trained ResNet50 model without pre-trained weights
    model = models.resnet50(pretrained=False)  # Load without pre-trained weights
    # 3. Freeze model parameters (if necessary, only if you also froze them during training)
    for param in model.parameters():
        param.requires_grad = False

    # 4. Replace the last fully connected layer (to match saved model architecture)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()  # Apply Sigmoid activation function
    )

    #resnet model name model.pth
    model_state = torch.load(os.path.join(settings.MODELS_PATH, 'adversarial_model1.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(model_state)


    # Load the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model

def load_swin_transformer_model():
    # 1. Load the Swin Transformer model without pre-trained weights
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=1)

    # 3. Freeze model parameters (if necessary, only if you also froze them during training)
    for param in model.parameters():
        param.requires_grad = False

    # 4. Unfreeze the last layer (to match saved model configuration)
    for param in model.head.parameters():
        param.requires_grad = True

    # 2. Load the saved model state
    model_state = torch.load(os.path.join(settings.MODELS_PATH, 'adversarial_swin_transformer_model.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

    return model

