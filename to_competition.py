import os

import numpy as np
import torch
from torchvision.io import read_image, write_png
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision import transforms
from tqdm import tqdm
import cv2

import parameters as pr
import utils

try:
    os.mkdir(pr.OUTPUT_DIR)
except FileExistsError:
    pass
to_tensor = transforms.ToTensor()

model = torch.load(f"{pr.CHECKPOINT_DIR}/{pr.CHECKPOINT_PATH}", map_location=pr.device)

with torch.no_grad():
    for file_name in tqdm(os.listdir(pr.TEST_DIR)):
        img = read_image(f"{pr.TEST_DIR}/{file_name}")
        original_size = [val for val in img.shape[:-1]] 
        resized = TF.resize(to_tensor(img).unsqueeze(0), pr.img_size) 
        pred = model(resized.to(pr.device)).detach().cpu() 
        
        resized_pred = TF.resize(pred, original_size)
        res = utils.argmax2img(resized_pred[0].argmax(dim=0))
        write_png(res, f"{pr.OUTPUT_DIR}/{file_name[:-5]}.png")


import pandas as pd


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


dir = pr.OUTPUT_DIR
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'soutput.csv', index=False)