import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from data import IMAGE_HEIGHT,IMAGE_WIDTH

class Conv_BatchNormalizattion(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Conv_BatchNormalizattion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)

class Residual(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Residual, self).__init__()
        self.Residual = nn.Sequential(
            Conv_BatchNormalizattion(input_channels=input_channels, output_channels=output_channels),
            Conv_BatchNormalizattion(input_channels=output_channels, output_channels=output_channels),
            Conv_BatchNormalizattion(input_channels=output_channels, output_channels=output_channels)
        )

    def forward(self, X):
        return self.Residual(X)

class Conv_Resdiual_Conv(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Conv_Resdiual_Conv, self).__init__()
        self.first_conv = Conv_BatchNormalizattion(input_channels=input_channels, output_channels=output_channels)
        self.residual = Residual(input_channels=output_channels, output_channels=output_channels)
        self.last_conv = Conv_BatchNormalizattion(input_channels=output_channels, output_channels=output_channels)

    def forward(self,x):
        res_first_conv = self.first_conv(x)
        res_residual = self.residual(res_first_conv)
        pixel_wise_added = res_first_conv + res_residual
        return self.last_conv(pixel_wise_added)

class DeConv_BatchNormalizattion(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DeConv_BatchNormalizattion, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(input_channels,output_channels, 2, 2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)

class FusionNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, layers_num_channels=[64, 128, 256, 512]):
        super(FusionNet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.out_channels = out_channels

        # down-sampling part of unet
        for num_channels in layers_num_channels:
            self.down.append(Conv_Resdiual_Conv(in_channels, num_channels))
            in_channels = num_channels

        # double conv between the down and up sides
        self.bottom = Conv_Resdiual_Conv(layers_num_channels[-1], 2 * layers_num_channels[-1])

        # up-sampling part of unet
        for num_channels in reversed(layers_num_channels):
            self.up.extend([
                # there were two args of 2,2
                DeConv_BatchNormalizattion(num_channels * 2, num_channels),
                Conv_Resdiual_Conv(num_channels , num_channels)
            ])

        self.final_layer = nn.Conv2d(layers_num_channels[0], out_channels, kernel_size=1)

        # pooling layer we're going to use
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        '''
            forward function returns per pixel un normalized scores - (befroe applying softmax)
        '''

        # skip connection we aim to append in the up part
        skip_connections = []

        # forward on the down part
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # forward on the bottom
        x = self.bottom(x)

        skip_connections.reverse()

        # forward on the up part
        for index in range(0, len(self.up), 2):
            x = self.up[index](x)
            skip_connection = (skip_connections[index // 2])

            # happens if we didn't start with dimensions divisible by 16
            if x.shape != skip_connection.shape:
                print(f'x shape is {x.shape}, skip_connection_shape is {skip_connection.shape}')
                print("start with dimensions divisible by 16, or think how to solve it")
                raise IndexError

            merge_x = torch.add(skip_connection, x)
            x = self.up[index + 1](merge_x)

        return self.final_layer(x)

    def predict_labels_from_scores(self,pred_scores):
        '''
            given class raw scores (un normalized) -> returns per pixel classification
        '''
        with torch.no_grad():
            if self.out_channels == 1:
                pred_proba = nn.Sigmoid()(pred_scores)
                pred_labels = (pred_proba > 0.5).float()
            else : 
                pred_proba = nn.Softmax(dim = 1)(pred_scores)
                pred_labels = torch.argmax(pred_proba, dim = 1).unsqueeze(1)

        return pred_labels

    def predict_mask(self, img_batch): 
        self.predict_labels_from_scores(self.forward(img_batch))

    def sliding_window_validation(self, img_batch, mask_batch = None, verbose = False):
        """
            mask_batch is none if this is a test image without a ground truth mask. 
            mask_batch is not none if this is a validation image with a ground truth mask. 
            
        """
        
        roi_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
        sw_batch_size = img_batch.shape[0]
        per_pixel_score_predictions = sliding_window_inference(img_batch, roi_size, sw_batch_size, self,  padding_mode='replicate', overlap=0, progress=verbose)
        return per_pixel_score_predictions
