import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RGBDataset(Dataset):

    def __init__(self, path, filesRGB, filesDose, dispTrainingData=False):

        self.rgb_path = os.path.join(path, filesRGB[0])
        self.rgb = normalize(np.load(self.rgb_path)).astype(np.float32)
        # self.rgb = od(np.load(self.rgb_path)).astype(np.float32)

        self.dose_path = os.path.join(path, filesDose[0])
        self.dose = np.load(self.dose_path).astype(np.float32)

        for i in range(len(filesRGB)-1):
            newrgbpath = os.path.join(path, filesRGB[i+1])
            rgbtmp = normalize(np.load(newrgbpath)).astype(np.float32)
            # rgbtmp = od(np.load(self.rgb_path)).astype(np.float32)
            self.rgb = np.concatenate((self.rgb, rgbtmp), axis=0)

            newdosepath = os.path.join(path, filesDose[i+1])
            dosetmp = np.load(newdosepath).astype(np.float32)
            self.dose = np.concatenate((self.dose, dosetmp), axis=0)

        if dispTrainingData:
            fig = plt.figure(figsize=(7, 14))
            ax1 = fig.add_subplot(211)
            ax1.imshow(self.dose, interpolation='nearest', cmap=cm.viridis)
            ax2 = fig.add_subplot(212)
            ax2.imshow(self.rgb[:,:,0], interpolation='nearest', cmap=cm.Greys)
            plt.show()

        self.dose = self.dose / self.dose.max()

    def __getitem__(self, index):
        dose_line = self.dose[index]
        # rgb_line = self.rgb[index].flatten()
        rgb_line = self.rgb[index]

        return rgb_line,  np.expand_dims(dose_line, axis=0)

    def __len__(self):
        return self.dose.shape[0]


class RGBDataloader:

    def __init__(self, path, fileRGB, fileDose,  batch_size, shuffle=True, dispTrainingData=False):
        self.dataset = RGBDataset(path, fileRGB, fileDose, dispTrainingData=dispTrainingData)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def load_data(self):
        return self.dataloader
    def __len__(self):
        return len(self.dataloader)


class RBGBDataset(Dataset):

    def __init__(self, path, filesRGB, filesDose, dispTrainingData=False):

        self.normValue = 1.6
        self.rgb_path = os.path.join(path, filesRGB[0])
        rgb = np.load(self.rgb_path).astype(np.float32)
        rb = rgb[:,:,0]/rgb[:,:,2]
        gb = rgb[:,:,1]/rgb[:,:,2]
        self.rbgb = np.zeros((rb.shape[0],rb.shape[1],2)).astype(np.float32)
        self.rbgb[:,:,0] = (rb / self.normValue)       # Normalisation? np.amax(rb)
        self.rbgb[:,:,1] = (gb / self.normValue)

        self.dose_path = os.path.join(path, filesDose[0])
        self.dose = np.load(self.dose_path).astype(np.float32)

        for i in range(len(filesRGB)-1):
            newrgbpath = os.path.join(path, filesRGB[i+1])
            rgb = np.load(newrgbpath).astype(np.float32)
            rb = rgb[:, :, 0] / rgb[:, :, 2]
            gb = rgb[:, :, 1] / rgb[:, :, 2]
            rbgbtmp = np.zeros((rb.shape[0], rb.shape[1], 2)).astype(np.float32)
            rbgbtmp[:, :, 0] = (rb / self.normValue)
            rbgbtmp[:, :, 1] = (gb / self.normValue)
            self.rbgb = np.concatenate((self.rbgb, rbgbtmp), axis=0)

            newdosepath = os.path.join(path, filesDose[i+1])
            dosetmp = np.load(newdosepath).astype(np.float32)
            self.dose = np.concatenate((self.dose, dosetmp), axis=0)

        if dispTrainingData:
            fig = plt.figure(figsize=(7, 14))
            ax1 = fig.add_subplot(211)
            ax1.imshow(self.dose, interpolation='nearest', cmap=cm.viridis)
            ax2 = fig.add_subplot(212)
            ax2.imshow(self.rbgb[:,:,0], interpolation='nearest', cmap=cm.Greys)
            plt.show()

        self.dose = self.dose / self.dose.max()

    def __getitem__(self, index):
        dose_line = self.dose[index]
        rgb_line = self.rbgb[index]

        return rgb_line,  np.expand_dims(dose_line, axis=0)

    def __len__(self):
        return self.dose.shape[0]


class RBGBDataloader:

    def __init__(self, path, fileRGB, fileDose,  batch_size, shuffle=True, dispTrainingData=False):
        self.dataset = RBGBDataset(path, fileRGB, fileDose, dispTrainingData=dispTrainingData)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataloader)


def normalize(x, maximum=65535):
    return x / (maximum / 2.) - 1


def od(x, maximum=65536):
    return np.log10(maximum/x)
