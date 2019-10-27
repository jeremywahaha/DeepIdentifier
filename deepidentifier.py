import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, zero_pad):
        super(DSConv, self).__init__()
        self.zp = nn.ZeroPad2d(zero_pad)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel, stride=stride, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.zp(x))))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class Funnel_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, zero_pad):
        super(DSIBlock, self).__init__()

        kernel1 = kernel
        kernel2 = (kernel[0] * 2, kernel[1])
        kernel3 = (kernel[0] * 3, kernel[1])

        zero_pad1 = zero_pad
        zero_pad2 = (zero_pad[0], zero_pad[1], zero_pad[2] + 3, zero_pad[3] + 3)
        zero_pad3 = (zero_pad[0], zero_pad[1], zero_pad[2] + 6, zero_pad[3] + 6)

        split_planes = int(out_planes / 4)

        self.conv1 = DSConv(in_planes, split_planes, kernel1, stride, zero_pad1)
        self.conv2 = DSConv(in_planes, split_planes, kernel2, stride, zero_pad2)
        self.conv3 = DSConv(in_planes, split_planes, kernel3, stride, zero_pad3)
        self.maxpool = nn.Sequential(
            nn.ZeroPad2d(zero_pad1),
            nn.MaxPool2d(kernel1, 1),
            nn.Conv2d(in_planes, split_planes, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.maxpool(x)

        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out

class Encoder(nn.Module):
    def __init__(self, input_shape=450, z_size=10):
        super(Encoder, self).__init__()

        self.mobileinception = nn.Sequential(
                Funnel_Block(1, 64, (6, 1), (1, 1), (0, 0, 2, 3)),
                Funnel_Block(64, 64, (6, 1), (5, 1), (0, 0, 2, 3)),
                Blockv1(64, 64, (6, 6), (1, 1), (2, 3, 2, 3)),
                Blockv1(64, 64, (6, 6), (2, 1), (2, 3, 2, 2)),
                Blockv1(64, 64, (6, 6), (1, 1), (2, 3, 2, 3)),
                Blockv1(64, 64, (6, 6), (2, 1), (0, 0, 2, 2))
            )
        self.fc = nn.Linear(int(input_shape / 120 * 64) , z_size)

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DeepIdentifier(nn.Module):
    def __init__(self, input_shape=450, filters=[32, 64, 128, 10]):
        super(DeepIdentifier, self).__init__()

        self.encoder = Encoder()
        self.input_shape = input_shape
        self.filters = filters

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, stride=(1, 1), padding=(1, 0)),

            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 1, 0)),
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, stride=(2, 1), padding=(1, 2)),

            nn.ReLU(),
            nn.ZeroPad2d((0, 0, 0, 0)),
            nn.ConvTranspose2d(filters[0], 1, kernel_size=(6, 1), stride=(3, 1), padding=(3, 0))
        )
        self.fc1 = nn.Linear(filters[3], int(25 * filters[2]))
        self.fc2 = nn.Linear(z_size, 2)

    def forward(self, x):
        z = self.encoder(x)
        out1 = self.fc2(z)

        out = self.fc1(z)
        out = out.view(out.size(0), self.filters[2], -1, 1)
        out2 = self.decoder(out)

        return out1, out2
