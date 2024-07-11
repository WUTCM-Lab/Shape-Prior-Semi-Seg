import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from utils.aug_function import FeatureNoiseDecoder, DropOutDecoder


def cat(x1, x2, x3=None, dim=1):
    if x3 == None:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        return x
    else:
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim)
        diffY = torch.tensor([x.size()[2] - x3.size()[2]])
        diffX = torch.tensor([x.size()[3] - x3.size()[3]])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        if transpose:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels,
                                #    in_channels // 4,
                                   out_channels,
                                #    in_channels // 4,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=False),
                # nn.BatchNorm2d(in_channels // 4),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load("pretrain/backbone/resnet34.pth"))

        if in_channels == 3:
            self.encoder1_conv = resnet.conv1
        else:
            self.encoder1_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_maxpool = self.maxpool(e1)

        e2 = self.encoder2(e1_maxpool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        return e1, e2, e3, e4, e5


class MyModel(nn.Module):
    def __init__(self,
                 num_classes=1,
                 in_channels=3
                 ):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        
        # seg-Decoder
        self.segDecoder5 = DecoderBlock(512, 512)
        self.segDecoder4 = DecoderBlock(512 + 256, 256)
        self.segDecoder3 = DecoderBlock(256 + 128, 128)
        self.segDecoder2 = DecoderBlock(128 + 64, 64)
        self.segDecoder1 = DecoderBlock(64 + 64, 64)

        self.segconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        # inpaint-Decoder
        
        self.inpDecoder5 = DecoderBlock(512, 512, transpose=True)
        self.inpDecoder4 = DecoderBlock(512 + 256, 256, transpose=True)
        self.inpDecoder3 = DecoderBlock(256 + 128, 128, transpose=True)
        self.inpDecoder2 = DecoderBlock(128 + 64, 64, transpose=True)
        self.inpDecoder1 = DecoderBlock(64 + 64, 64, transpose=True)

        self.inpSideout5 = SideoutBlock(512, 1)
        self.inpSideout4 = SideoutBlock(256, 1)
        self.inpSideout3 = SideoutBlock(128, 1)
        self.inpSideout2 = SideoutBlock(64, 1)

        self.inpconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        self.dropout = DropOutDecoder()      

    def forward(self, x):

        ori = x
        bs, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        """Seg-branch"""
        e1, e2, e3, e4, e5 = self.encoder(x)

        d5 = self.segDecoder5(e5)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))
        
        mask = self.segconv(d1)
        mask = torch.sigmoid(mask)
        mask_binary = (mask > 0.5)
        mask_binary = mask_binary.float()

        # inpe1, inpe2, inpe3, inpe4, inpe5 = self.encoder(x)

        e1, e2, e3, e4, e5 = self.encoder(x)

        # inpe1, inpe2, inpe3, inpe4, inpe5 = self.encoder(x)
        inpe1, inpe2, inpe3, inpe4, inpe5 = e1, e2, e3, e4, self.dropout(e5)
        
        d5 = self.segDecoder5(e5)
        d4 = self.segDecoder4(cat(d5, e4))
        d3 = self.segDecoder3(cat(d4, e3))
        d2 = self.segDecoder2(cat(d3, e2))
        d1 = self.segDecoder1(cat(d2, e1))
        
        mask = self.segconv(d1)
        mask = torch.sigmoid(mask)

        inpd5 = self.inpDecoder5(inpe5)
        inpimg5 = self.inpSideout5(inpd5)
        inpimg5 = torch.sigmoid(inpimg5)
        inpd4 = self.inpDecoder4(cat(inpd5, inpe4))
        inpimg4 = self.inpSideout4(inpd4)
        inpimg4 = torch.sigmoid(inpimg4)
        inpd3 = self.inpDecoder3(cat(inpd4, inpe3))
        inpimg3 = self.inpSideout3(inpd3)
        inpimg3 = torch.sigmoid(inpimg3)
        inpd2 = self.inpDecoder2(cat(inpd3, inpe2))
        inpimg2 = self.inpSideout2(inpd2)
        inpimg2 = torch.sigmoid(inpimg2)
        inpd1 = self.inpDecoder1(cat(inpd2, inpe1))
        preboud = self.inpconv(inpd1)
        preboud = torch.sigmoid(preboud)
        return mask, preboud, inpimg2, inpimg3, inpimg4, inpimg5, mask_binary
        # return mask, preboud, mask_binary


