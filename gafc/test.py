import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gafc.dataloader import RGBDataloader
from gafc.model import ModelRGB

if __name__ == '__main__':

    checkpoints_dir = 'checkpoints/'
    name = 'gaf_6'
    expr_dir = os.path.join(checkpoints_dir, name)

    root_training = 'data'
    # filesDose = ['lot02282002_FFFstep-Center_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Center_RGBArray.npy']
    # filesDose = ['lot02282002_FFFstep-Right_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Right_RGBArray.npy']
    # filesDose = ['lot02282002_FFFstep-Left_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Left_RGBArray.npy']

    # filesDose = ['lot02282001_clinacStar-Center_regDose.npy']
    # filesRGB = ['lot02282001_clinacStar-Center_RGBArray.npy']
    # filesDose = ['lot02282001_clinacStar-Left_regDose.npy']
    # filesRGB = ['lot02282001_clinacStar-Left_RGBArray.npy']

    # filesDose = ['lot02282001_ClinacStep-Center_regDose.npy']
    # filesRGB = ['lot02282001_ClinacStep-Center_RGBArray.npy']

    # filesDose = ['lot02282002_HalcyonWedge90-Left_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge90-Left_RGBArray.npy']
    # filesDose = ['lot02282002_HalcyonWedge90-Center_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge90-Center_RGBArray.npy']
    # filesDose = ['lot02282002_HalcyonWedge90-Right_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge90-Right_RGBArray.npy']

    filesDose = ['lot02282002_HalcyonWedge-Left_regDose.npy']
    filesRGB = ['lot02282002_HalcyonWedge-Left_RGBArray.npy']
    # filesDose = ['lot02282002_HalcyonWedge-Center_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge-Center_RGBArray.npy']
    # filesDose = ['lot02282002_HalcyonWedge-Right_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge-Right_RGBArray.npy']


    batch_size = 1 #a ne plus changer !

    dataloader = RGBDataloader(root_training, filesRGB, filesDose, batch_size, shuffle=False)
    dataset = dataloader.load_data()

    model = ModelRGB(expr_dir, batch_size=batch_size, nf=64)

    checkpoint = torch.load(os.path.join(expr_dir, "latest"))
    model.net.load_state_dict(checkpoint['net'])

    doseimg = np.load(os.path.join(root_training, filesDose[0]))
    max = doseimg.max()
    calcdoseimg = np.zeros((doseimg.shape[0], doseimg.shape[1]))

    with torch.no_grad():
        i: int = 0
        for rgb, dose in dataset:
            print(i)
            dose = model.net(rgb.transpose(1,2))
            for j in range(batch_size):
                if i*batch_size+j < doseimg.shape[0]:
                    calcdoseimg[i*batch_size+j, :] = dose[j, :]*max
            i = i+1
            #print(model.net(rgb))


    linemin = 70
    step = 20
    nboflines = 7

    colors = ["steelblue", "olive", "firebrick", "darkslategray", "indigo", "deepskyblue", "lime", "darkorange",
              "gold", "turquoise", "darkgreen", "orangered", "fuchsia"]
    fig = plt.figure(figsize=(7,14))
    ax1 = fig.add_subplot(211)
    #ax1.imshow(doseimg, interpolation='bilinear', cmap=cm.viridis)
    ax1.imshow(calcdoseimg, interpolation='nearest', cmap=cm.viridis)
    ax2 = fig.add_subplot(212)
    for i in range(nboflines):
        y = linemin + step*i
        ax2.plot(np.arange(doseimg.shape[1]), doseimg[y], color=colors[i],linewidth=1, linestyle="solid")
        ax2.plot(np.arange(doseimg.shape[1]), calcdoseimg[y], color=colors[i],linewidth=1, linestyle="dashed")
    plt.show()

