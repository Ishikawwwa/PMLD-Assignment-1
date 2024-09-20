import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image

class ButterCLF(nn.Module):
    def __init__(self, num_classes):
        super(ButterCLF, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ModelRunner():
    def __init__(self) -> None:
        self.model = ButterCLF(75)
        self.model.load_state_dict(torch.load("api/best.pt", weights_only=True, map_location=torch.device('cpu')))
        self.model.eval()

        print("Model loaded successfully")

        self.transform = v2.Compose([
            v2.Resize((128,128)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    
    def get_prediction(self, img_path, classes_path):
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img)
        
        _, predicted_class = torch.max(output, 1)
        
        classes = []
        with open(classes_path) as f:
            classes = [line.rstrip() for line in f]

        return classes[predicted_class.item()]