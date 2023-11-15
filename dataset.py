import os

import torch
from torchvision.io import read_image
from torchvision.transforms import functional as TF


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory:str, transform=None, standard_size = (608, 800), train = True):
        '''Param:
        `dir` (string): Path to the directory containing samples. This directory cannot contain any other files.
        `name` (string, optional): Name of the dataset
        `transform` (callable, optional): The transformation to be applied to a sample'''

        self.__train = train
        self.transform = transform
        self.__standard_size = standard_size # Sketch images to one standard size. 1280x1024 by default.
        self.__list_samples__ = [] # A list containing samples' filenames.
        self.__list_gt__ = [] # A list containing grouth truth images' filenames.
        
        try:
            if self.__train:
                list_files = os.listdir(f"{directory}/train/train")
                for file_name in list_files: # Sample and ground truth pair should have the same name.
                    self.__list_samples__.append(f'{directory}/train/train/{file_name}')
                    self.__list_gt__.append(f'{directory}/train_gt/train_gt/{file_name}')
            else:
                list_files = os.listdir(f"{directory}/test/test")
                for file_name in list_files:
                    self.__list_samples__.append(f'{directory}/test/test/{file_name}')
        except FileNotFoundError:
            err_msg = f"Directory {directory} does not exist!"
            raise FileNotFoundError(err_msg)
    
    def __len__(self):
        return len(self.__list_samples__)

    def __getitem__(self, idx):
        sample = read_image(self.__list_samples__[idx]) / 255
        if self.__train:
            gt = read_image(self.__list_gt__[idx]) / 255
        else:
            gt = torch.zeros(size=sample.shape)
        if self.transform:
            sample, gt = self.transform(sample, gt)
        sample = TF.resize(sample, self.__standard_size)
        gt = TF.resize(gt, self.__standard_size)
        gt = ((gt - 0.8) > 0).float()
        # To class label
        red = gt[0]
        green = gt[1]
        background = 1 - (red + green) # No red or no green --> background = 1
        gt = green + background*2 # gt[..] = 0 means red, gt[..] = 1 means green, gt[..] = 2 means background
        return sample, gt.long()