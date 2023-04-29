import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.bn3(self.conv3(x) + out)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoders = nn.ModuleList([
            EncoderBlock(3, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 32),
            EncoderBlock(32, 16),
            EncoderBlock(16, 3),
        ])

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i < len(self.encoders) - 1:
                x = self.relu(x)
        return self.out(x)
