import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image as PILImage

trained_model = None
class_names = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo', 'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']


class freshnessClassifierResNet(nn.Module):
    def __init__(self, num_classes=16,dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze the pre-trained weights
        
        ## unfreeze the last layer for learning new features
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return x


def predict(image):

    if isinstance(image, str):
        image = PILImage.open(image).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ## converting the image into batch of 1
    ### (3,224,224) -> (1,3,224,224)
    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    # Load the trained model only the first time
    if trained_model is None:
        trained_model = freshnessClassifierResNet()
        trained_model.load_state_dict(torch.load("model\saved_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
