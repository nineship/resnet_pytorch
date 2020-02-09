import torch
import torch.nn as nn
import torchvision
import numpy as np

import torch.nn.functional as F
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=512, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class pose_Net(nn.Module):
        def __init__(self):
                super(pose_Net,self).__init__()
                self.input = torch.nn.Linear(4, 256)
                self.hidden1 = torch.nn.Linear(256,512)
                self.hidden2 = torch.nn.Linear(512,512)
                self.hidden3 = torch.nn.Linear(512, 512)
                self.out = torch.nn.Linear(512,512)
        def forward(self,x):
            x = torch.relu(self.input(x.cuda()))
            x = torch.relu(self.hidden1(x))
            x = torch.relu(self.hidden2(x))
            x = torch.relu(self.hidden3(x))
            return x
def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])
class ActionDetect_Net(nn.Module):
        def __init__(self):
                super(ActionDetect_Net,self).__init__()
                self.resnet = ResNet50()
                self.posenet = pose_Net()

                self.hidden1 = torch.nn.Linear(1024, 512)
                self.hidden2 = torch.nn.Linear(512, 512)
                self.hidden3 = torch.nn.Linear(512, 512)
                self.out = torch.nn.Linear(512, 3)

        def forward(self,pose,pic):
                x_pose = self.posenet(pose)
                x_resnet = self.resnet(pic)
                #print(x_pose.shape,x_resnet.shape)
                x = torch.cat((x_pose,x_resnet),1)
                x = torch.relu(self.hidden1(x))
                #x.retain_grad()
                #print(x.grad)
                x = torch.relu(self.hidden2(x))
                x = torch.relu(self.hidden3(x))
                x = self.out(x)
                x = F.softmax(x,dim=1)
                return x
if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ActionDetect_Net()
    pic = torch.randn(1, 3, 224, 224)
    pose = torch.randn(1,4)
    pose = pose.float()
    out = model(pose,pic)
