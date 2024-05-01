import torch
from torchvision import transforms
from PIL import Image
from .apps import PredAppConfig


# Load the trained models
resnet50 = PredAppConfig.resnet50
swin_transformer = PredAppConfig.swin_transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path)

    # Preprocess the image
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make predictions using the models
    with torch.no_grad():
        img = img.to(device)
        resnet50_op = resnet50(img)
        resnet50_pred = (resnet50_op > 0.5).float().item()

        swin_op = swin_transformer(img)
        swin_pred = (swin_op > 0.5).float().item()
    
    return resnet50_pred, swin_pred
    