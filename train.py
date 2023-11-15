import logging

logging.basicConfig(level=logging.ERROR)

from datetime import datetime, timedelta, timezone

import torch
import wandb

import parameters as pr
import utils
from dataset import ImageDataset
from model import UNet

train_img = ImageDataset(directory=pr.INPUT_DIR, train=True, transform=utils.transform, standard_size=pr.img_size)
train_dataloader = torch.utils.data.DataLoader(train_img,
                                               batch_size=pr.batch_size,
                                               shuffle=True,
                                               num_workers=4,)
model = UNet([32, 64, 128, 256, 512], 3, )
model.to(pr.device)

wandb.login()
wandb.init(
    project="IntroToDLSegmentation",
    name=f'{pr.pretrain}_weightedCE_{datetime.now(tz=timezone(timedelta(hours=7)))}',
    config={"learning_rate":pr.learning_rate,
            "architecture": "UNet",
            "dataset": "BK NeoPolyp",
            "epochs": pr.num_epochs,
           },
    settings=wandb.Settings(start_method='fork')
)
utils.train(train_dataloader, model, num_epochs = pr.num_epochs, learning_rate = pr.learning_rate, loss_func= pr.loss_func, pretrain_name=pr.pretrain, end_learning_rate = pr.end_learning_rate)
wandb.finish()