import torch
import torchvision
import torchvision.transforms as transforms

Debug = True

# Setting the device for the Training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if Debug:
    if torch.cuda.is_available():
        print('the Training is on GPU :)')
    else:
        print('the Training is on CPU :(')


# Setting training's parameters
learningRate = 0
epochs = 0
batchSize = 0
optimizer = 'null'
