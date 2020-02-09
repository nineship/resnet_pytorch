# PyTorch中的torchvision里有很多常用的模型，可以直接调用：
import torchvision.models as models
from action_net import ActionDetect_Net 
from torchvision import datasets, models, transforms
import torch
from PIL import Image

test_valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth')

with torch.no_grad():
    image = Image.open("./test.jpg").convert('RGB')   # use skitimage
    image = image.resize((224, 224), Image.BILINEAR)
    image = test_valid_transforms(image)
    pos = torch.ones(4).cuda()
    inputs = image.to(device)
    inputs = inputs.unsqueeze(0)
    pos = pos.unsqueeze(0)
    outputs = model(pos,inputs)
    out = torch.max(outputs.data, 1)
    print(out)
