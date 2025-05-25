import torch
from torchvision.models import resnet18
from torch import nn

class ResnetModule(nn.Module):
    def __init__(self, config):
        super(ResnetModule, self).__init__()
        
        self.device = config.device
        self.model = resnet18(weights=None)
        # 修改网络第一层为 3*3卷积核, padding=1, stride=1
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, config.num_classes, bias=False)
        self.model = self.model.to(self.device)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    @torch.no_grad
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=False)
    # 修改网络第一层为 3*3卷积核, padding=1, stride=1
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10, bias=False)
    model = model.to('cuda:0')
    print(model)
    
    with torch.no_grad():
        x = torch.randn((5, 1, 28, 28)).to('cuda:0')
        y = model(x)
        print(model, '\n最终结果为:', y)
    