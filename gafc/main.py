import os

from dataloader import RGBDataloader
from dataloader import RBGBDataloader

from model import ModelRGB
from model import ModelRBGB

if __name__ == '__main__':
    checkpoints_dir = 'checkpoints/'
    name = 'gaf_6'
    expr_dir = os.path.join(checkpoints_dir, name)

    root_training = 'data'
    # filesDose = ['lot02282002_FFFstep-Left_regDose.npy',
    #              'lot02282002_FFFstep-Center_regDose.npy',
    #              'lot02282002_FFFstep-Right_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Left_RGBArray.npy',
    #             'lot02282002_FFFstep-Center_RGBArray.npy',
    #             'lot02282002_FFFstep-Right_RGBArray.npy']
    # filesDose = ['lot02282002_FFFstep-Center_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Center_RGBArray.npy']

    # filesDose = ['lot02282002_FFFstep-Left_regDose.npy',
    #              'lot02282002_FFFstep-Center_regDose.npy',
    #              'lot02282002_FFFstep-Right_regDose.npy',
    #              'lot02282001_ClinacStep-Center_regDose.npy']
    # filesRGB = ['lot02282002_FFFstep-Left_RGBArray.npy',
    #             'lot02282002_FFFstep-Center_RGBArray.npy',
    #             'lot02282002_FFFstep-Right_RGBArray.npy',
    #             'lot02282001_ClinacStep-Center_RGBArray.npy']

    # filesDose = ['lot02282002_HalcyonWedge90-Left_regDose.npy',
    #              'lot02282002_HalcyonWedge90-Center_regDose.npy',
    #              'lot02282002_HalcyonWedge90-Right_regDose.npy',
    #              'lot02282002_HalcyonWedge-Left_regDose.npy',
    #              'lot02282002_HalcyonWedge-Center_regDose.npy',
    #              'lot02282002_HalcyonWedge-Right_regDose.npy']
    # filesRGB = ['lot02282002_HalcyonWedge90-Left_RGBArray.npy',
    #             'lot02282002_HalcyonWedge90-Center_RGBArray.npy',
    #             'lot02282002_HalcyonWedge90-Right_RGBArray.npy',
    #             'lot02282002_HalcyonWedge-Left_RGBArray.npy',
    #             'lot02282002_HalcyonWedge-Center_RGBArray.npy',
    #             'lot02282002_HalcyonWedge-Right_RGBArray.npy']

    filesDose = ['lot02282002_HalcyonWedge90-Center_regDose.npy']
    filesRGB = ['lot02282002_HalcyonWedge90-Center_RGBArray.npy']

    batch_size = 1

    expr_dir = os.path.join(checkpoints_dir, name)

    # dataloader = RGBDataloader(root_training, filesRGB, filesDose,
    #                            batch_size, dispTrainingData=True)
    dataloader = RBGBDataloader(root_training, filesRGB, filesDose,
                               batch_size, dispTrainingData=True)
    dataset = dataloader.load_data()

    # model = ModelRGB(expr_dir, batch_size=batch_size, nf=64, niter=150, niter_decay=150)
    model = ModelRBGB(expr_dir, batch_size=batch_size, nf=64, niter=150, niter_decay=150)
    model.train(dataset)

    # f = dose.detach().numpy()
    # t = dose_predicted.detach().numpy()
