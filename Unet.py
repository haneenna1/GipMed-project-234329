import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from data import IMAGE_HEIGHT,IMAGE_WIDTH


class DoubleConv(nn.Module):

    def __init__(self, input_channels, output_channels, dropout= 0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, X):
        return self.conv(X)


class Unet(nn.Module):

    # NOTE - always start with an input dims divisible by 16 because we do 4 times downsampling by factor 2
    def __init__(self, in_channels=3, out_channels=2, layers_num_channels=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.out_channels = out_channels

        # down-sampling part of unet
        for num_channels in layers_num_channels:
            self.down.append(DoubleConv(in_channels, num_channels))
            in_channels = num_channels

        # double conv between the down and up sides
        self.bottom = DoubleConv(layers_num_channels[-1], 2 * layers_num_channels[-1])

        # up-sampling part of unet
        for num_channels in reversed(layers_num_channels):
            self.up.extend([
                nn.ConvTranspose2d(num_channels * 2, num_channels, 2, 2),
                DoubleConv(2 * num_channels, num_channels)
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

            concat_x = torch.cat((x, skip_connection), dim=1)
            x = self.up[index + 1](concat_x)

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
        per_pixel_score_predictions = sliding_window_inference(img_batch, roi_size, sw_batch_size, self,  padding_mode='reflect', overlap=0, progress=verbose)
        return per_pixel_score_predictions

