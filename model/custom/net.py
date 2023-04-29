import torch.nn as nn
import torch


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x), x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.upbn = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(in_ch + out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, features):
        x = self.relu(self.upbn(self.upconv(x)))
        gated_x = self.relu(self.bn1(self.conv1(torch.cat([x, features], dim=1))))
        x = self.bn2(self.conv2(x + gated_x))
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoders = nn.ModuleList([
            EncoderBlock(3, 64),  # N, 64, 128, 128
            EncoderBlock(64, 128),  # N, 128, 64, 64
            EncoderBlock(128, 256),  # N, 256, 32, 32
        ])

        self.decoders = nn.ModuleList([
            DecoderBlock(256, 128),  # N, 128, 64, 64
            DecoderBlock(128, 64),  # N, 64, 128, 128
            DecoderBlock(64, 3),  # N, 3, 256, 256
        ])

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc_features = []
        for i in range(len(self.encoders)):
            x, features = self.encoders[i](x)
            enc_features.append(features)
        enc_features = enc_features[::-1]
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, enc_features[i])
            if i < len(self.decoders) - 1:
                x = self.relu(x)
        return self.out(x)
