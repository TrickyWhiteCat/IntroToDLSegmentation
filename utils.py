import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import math
import wandb
from datetime import datetime, timezone, timedelta
import parameters as pr

def argmax2img(arr):
    if len(arr.shape) == 4:
        return torch.stack([argmax2img(img) for img in arr])
    img = torch.zeros(3, arr.shape[-2], arr.shape[-1])
    red = arr == 0
    green = arr == 1
    img[0, :, :] = red
    img[1, :, :] = green
    return img.float()

def transform(img, gt, random_seed: int = None):
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    transformation = transforms.Compose(*[transforms.RandomPerspective(p = 1),
                                          transforms.RandomRotation(degrees = math.pi / 4,expand = True),])
    stacked = torch.stack(tensors = [img, gt], dim = 0)
    stacked = transformation(stacked)
    img, gt = stacked[0, :, :], stacked[1, :, :]
    img = transforms.ColorJitter()(img)
    return img, gt

def train(dataloader, model, num_epochs = 100, learning_rate = 10**-3, loss_func = None, pretrain_name="scratch", start=0, end_learning_rate = None, report_step = 50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if end_learning_rate is not None:
        expo = (end_learning_rate / learning_rate) ** (1/num_epochs)
    else:
        expo = 1
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = expo)
    
    accu_loss = 0
    for e in range(1 + start, 1 + start+num_epochs):
        for idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(pr.device), y.to(pr.device)
            pred = model(x)
            y = TF.center_crop(y, [pred.shape[-2], pred.shape[-1]])
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {e}/{num_epochs} --- Batch {idx + 1}/{len(dataloader)} --- Loss {loss.item():.4f}", end='\r')
            accu_loss+= loss.detach()
            if ((idx + 1)*dataloader.batch_size) % report_step == 0:
                wandb.log({"Loss": accu_loss.item() / report_step})
                accu_loss = 0
                table = wandb.Table(columns=["Predict", "Target"])
                table.add_data(wandb.Image(pred[0]),
                               wandb.Image(argmax2img(y[0])))
                wandb.log({f"Comparision:": table})
        scheduler.step()
        
        checkpoint_path = f"{pretrain_name}_{datetime.now(tz=timezone(timedelta(hours=7)))}_{e}.pth"
        torch.save({"model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict()}, checkpoint_path)