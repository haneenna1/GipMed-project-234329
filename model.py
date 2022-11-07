import torch
import torch.nn as nn

class Unet(nn.Module):

    # TODO: AMIR is there any need to specify the padding == 1?
    # TODO: AMIR specifying RGB in channels?
    # TODO: AMIR check the dimensions for sanity issues
    # TODO: AMIR check the cropping that is happening while concatenating,what should we do with it?
    # TODO: AMIR check the non-even pictures dimensions if there any problen we shall encounter
    # TODO : AMIR how to initialize the weight ? the paper suggests from gaussian distribution (see UNET)

    #TODO : AMIR this model is without any pre-trained weights
    # suggested models as the papers gives:
    #   1- ResNet-V2 based
    #   2- DenseNet-V2 based
    #   3- maybe Nested unet

    # TODO : AMIR adding batchnorm and checking if there is a bias within the weight?
    # TODO : AMIR adding dropout layers

    def __init__(self):
        super().__init__()

        # The enumeration of the different layers is based on the depth
        # for example decoderDoubleConv4 at depth = 4 in the U - arch
        # The double convolutions within the Encoder path
        self.encoderDoubleConv1 = self.doubleConv(1, 64)
        self.encoderDoubleConv2 = self.doubleConv(64, 128)
        self.encoderDoubleConv3 = self.doubleConv(128, 256)
        self.encoderDoubleConv4 = self.doubleConv(256, 512)
        # The last sole convolution in the Encoder path
        self.encoderConv5 = nn.Conv2d(512, 1024, kernel_size=(3, 3))
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # The double convolutions within the Decoder path
        self.decoderConv5 = nn.Conv2d(1024, 1024, kernel_size=(3, 3))
        self.decoderDoubleConv4 = self.doubleConv(1024, 512)
        self.decoderDoubleConv3 = self.doubleConv(512, 256)
        self.decoderDoubleConv2 = self.doubleConv(256, 128)
        self.decoderDoubleConv1 = self.doubleConv(128, 64)

        # The last convolutions 1X1 within the Decoder path
        self.decoderConv1 = nn.Conv2d(64, 2, kernel_size=(1, 1))

        # The up-conv within the Decoder path
        self.decoderDeConv4 = nn.ConvTranspose2d(1024, 512, kernel_size=(2,2))
        self.decoderDeConv3 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2))
        self.decoderDeConv2 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2))
        self.decoderDeConv1 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2))

        # List for the concatenations tensors
        self.concatenationsList = []

    def forward(self, x):

        # The Encoding path
        x1 = self.encoderDoubleConv1(x)
        self.concatenationsList.append(x1)
        x2 = self.encoderDoubleConv2(self.maxPool(x1))
        self.concatenationsList.append(x2)
        x3 = self.encoderDoubleConv3(self.maxPool(x2))
        self.concatenationsList.append(x3)
        x4 = self.encoderDoubleConv4(self.maxPool(x3))
        self.concatenationsList.append(x4)

        x5 = self.encoderConv5(self.maxPool(x4))
        # The Decoding path
        x6 = self.decoderDeConv4(self.decoderConv5(x5))

        # The Decoding path
        # TODO: AMIR what the dim of the channel concatenation, modilfy the dim=1!
        x7 = self.decoderDoubleConv4(torch.concat([self.concatenationsList.pop(), x5], dim=1))
        x8 = self.decoderDoubleConv3(self.decoderDeConv3(x7))
        x9 = self.decoderDoubleConv2(self.decoderDeConv2(x8))
        x10 = self.decoderDoubleConv1(self.decoderDeConv1(x9))

        outputSegmentationMap = self.decoderConv1(x10)

        return outputSegmentationMap

    @staticmethod
    def doubleConv(input_channels, output_channels) -> nn.Sequential:
        doubleConv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        )
        return doubleConv



