import logging

logging.basicConfig(level=logging.ERROR)

from datetime import datetime, timedelta, timezone

import torch
import wandb

import parameters as pr
import utils
from dataset import ImageDataset
from model import UNet

def main():
    train_img = ImageDataset(directory=pr.INPUT_DIR, train=True, transform=utils.transform, standard_size=pr.img_size)
    train_set, vali_set = torch.utils.data.random_split(train_img, pr.split_ratio)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=pr.batch_size, shuffle=True, num_workers=pr.num_workers,)
    vali_dataloader = torch.utils.data.DataLoader(vali_set, batch_size=pr.batch_size, shuffle=True, num_workers=pr.num_workers,)
    model = UNet([16, 32, 64, 128, 256], 3, )
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
    )
    utils.train(train_dataloader, model, num_epochs = pr.num_epochs, learning_rate = pr.learning_rate, loss_func= pr.loss_func, pretrain_name=pr.pretrain, end_learning_rate = pr.end_learning_rate, report_step = pr.report_step, vali_dataloader=vali_dataloader)
    wandb.finish()

if __name__ == '__main__':
    main()