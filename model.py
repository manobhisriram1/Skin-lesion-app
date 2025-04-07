import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()

        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        # Pretrained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Identity()

        # Pretrained EfficientNet
        self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        self.efficientnet._fc = nn.Identity()

        # Dummy input to get feature sizes
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            cnn_out = self.cnn(sample_input)
            resnet_out = self.resnet(sample_input)
            efficient_out = self.efficientnet(sample_input)
            concat_size = cnn_out.shape[1] + resnet_out.shape[1] + efficient_out.shape[1]

        # Final classifier
        self.fc = nn.Linear(concat_size, 7)

    def forward(self, x):
        cnn_out = self.cnn(x)
        resnet_out = self.resnet(x)
        efficient_out = self.efficientnet(x)
        combined = torch.cat((cnn_out, resnet_out, efficient_out), dim=1)
        return self.fc(combined)

# Function to load model with weights
def load_model(model_path, device):
    model = EnsembleModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
