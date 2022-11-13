import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import Unet
from dataset import ThumbnailsDataset

# Setting training's parameters
learningRate = 0
epochs = 0
batchSize = 0
validationRatio = 0


class Train:
    def __init__(self, path):
        self.path = path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Unet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.lossFunction = None
        ImageDir = None
        maskDir = None
        self.dataset = ThumbnailsDataset(ImageDir, maskDir)

    def train(self):
        validationNum = int(len(self.dataset) * validationRatio)
        trainNum = len(self.dataset) - validationNum
        trainSet, valSet = random_split(self.dataset, [trainNum, validationNum])

        trainLoader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, num_workers=2)
        validationLoader = DataLoader(dataset=valSet, batch_size=batchSize, shuffle=True, num_workers=2)

        while True:
            # training epochs
            for epoch in range(1, epochs):
                torch.cuda.empty_cache()
                self.model.train()
                with tqdm(trainLoader, unit="batch") as epoch:
                    trainLoss = []
                    torch.cuda.empty_cache()
                    for i, inputs in enumerate(epoch):
                        image, mask, _ = inputs
                        image = image.to(self.device)
                        mask = mask.to(self.device)
                        epoch.set_description(f"Training Epoch {epoch}")
                        self.optimizer.zero_grad()
                        output = self.model.forward(image)
                        loss = self.lossFunction(output, mask)
                        loss.backward()
                        self.optimizer.step()

                        loss_value = loss.item()
                        trainLoss.append(loss_value)
            # validation epochs
            for epoch in range(1, epochs):

                torch.cuda.empty_cache()
                self.model.eval()
                with torch.no_grad():
                    with tqdm(validationLoader, unit="batch") as epoch:
                        validationLoss = []
                        torch.cuda.empty_cache()
                        for i, inputs in enumerate(epoch):
                            image, mask, _ = inputs
                            image = image.to(self.device)
                            mask =  mask.to(self.device)
                            epoch.set_description(f"Validation Epoch {epoch}")
                            self.optimizer.zero_grad()
                            output = self.model.forward(image)
                            loss = self.lossFunction(output, mask)
                            loss.backward()
                            self.optimizer.step()

                            loss_value = loss.item()
                            validationLoss.append(loss_value)
        #should check the losses and stop when we get the desired result