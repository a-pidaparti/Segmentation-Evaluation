import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
resnet = torchvision.models.resnet.resnet50(pretrained=True)


class SegNet(nn.Module):

    def __init__(self, in_channels, out_channels, BN_momentum=0.5):
        super(SegNet, self).__init__()


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers


        self.upsample0 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, 1024, kernel_size=1, stride=1)
        )

        self.Conv0 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.BN0 = nn.BatchNorm2d(1024, momentum=BN_momentum)
        self.Conv11 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.BN11 = nn.BatchNorm2d(1024, momentum=BN_momentum)
        self.Conv12 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.BN12 = nn.BatchNorm2d(1024, momentum=BN_momentum)




        self.upsample1 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(1024, 512, kernel_size=1, stride=1)
            )

        self.Conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.Conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.Conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(512, momentum=BN_momentum)


        self.upsample2 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(512, 256, kernel_size=1, stride=1)
        )

        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BN4 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.Conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BN5 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.Conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BN6 = nn.BatchNorm2d(256, momentum=BN_momentum)



        self.upsample3 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(256, 128, kernel_size=1, stride=1)
        )

        self.Conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BN7 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.Conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BN8 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.Conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BN9 = nn.BatchNorm2d(128, momentum=BN_momentum)



        self.upsample4 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(128, 64, kernel_size=1, stride=1)
        )

        self.Convx = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNx = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.Convy = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNy = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.Convz = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.BNz = nn.BatchNorm2d(out_channels, momentum=BN_momentum)


        self.softmax = nn.Softmax(dim=out_channels)






    def forward(self, x):



        #DECODE LAYERS
        #Stage 5
        #print(x.shape)
        x = self.upsample0(x)
        x = f.relu(self.BN0(self.Conv0(x)))
        #x = self.Conv0(x)
        #x = self.BN0(x)
        x = f.relu(self.BN11(self.Conv11(x)))
        #x = self.Conv11(x)
        #x = self.BN11(x)
        x = f.relu(self.BN12(self.Conv12(x)))
        #x = self.Conv12(x)
        #x = self.BN12(x)


        #Stage 4
        #print(x.shape)
        x = self.upsample1(x)
        x = f.relu(self.BN1(self.Conv1(x)))
        #x = self.Conv1(x)
        #x = self.BN1(x)
        x = f.relu(self.BN2(self.Conv2(x)))
        #x = self.Conv2(x)
        #x = self.BN2(x)
        x = f.relu(self.BN3(self.Conv3(x)))
        #x = self.Conv3(x)
        #x = self.BN3(x)

        #Stage 3
        #print(x.shape)
        x = self.upsample2(x)
        x = f.relu(self.BN4(self.Conv4(x)))
        #x = self.Conv4(x)
        #x = self.BN4(x)
        x = f.relu(self.BN5(self.Conv5(x)))
        #x = self.Conv5(x)
        #x = self.BN5(x)
        x = f.relu(self.BN6(self.Conv6(x)))
        #x = self.Conv6(x)
        #x = self.BN6(x)



        #Stage 2
        #print(x.shape)
        x = self.upsample3(x)
        x = f.relu(self.BN7(self.Conv7(x)))
        #x = self.Conv7(x)
        #x = self.BN7(x)
        x = f.relu(self.BN8(self.Conv8(x)))
        #x = self.Conv8(x)
        #x = self.BN8(x)
        x = f.relu(self.BN9(self.Conv9(x)))
        #x = self.Conv9(x)
        #x = self.BN9(x)

        #Stage 1
        #print(x.shape) 

        x = self.upsample4(x)
        x = f.relu(self.BNx(self.Convx(x)))
        #x = self.Convx(x)
        #x = self.BNx(x)
        x = f.relu(self.BNy(self.Convy(x)))
        #x = self.Convy(x)
        #x = self.BNy(x)
        x = f.relu(self.BNz(self.Convz(x)))
        #x = self.Convz(x)
        #x = self.BNz(x)

        x = self.softmax(x)

        #print(x.shape)
        return x

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )

        self.conv_block_1 = ConvBlock(out_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)
        self.conv_block_3 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=3):
        super().__init__()

        #ENCODE WITH RESNET50
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        self.segnet = SegNet(2048, n_classes)


    def forward(self, x, with_output_feature_map=False):

        #ENCODE WITH RESNET50
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x
            #print(x.shape)

        #print(x.shape)
        x = self.bridge(x)
        #print(x.shape)
        x = self.segnet(x)
        return x

