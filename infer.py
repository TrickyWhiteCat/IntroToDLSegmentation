from torchvision import io
import torch
import argparse
import utils
from model import *

def main():
    parser = argparse.ArgumentParser(description='Model inference')
    parser.add_argument('-c', '--checkpoint', type=str, help='Path to checkpoint.')
    parser.add_argument('-i', '--input', type=str, help='Path to input image.')
    parser.add_argument('-o', '--output', type=str, help='Path to output image. Default to <input image name>_out.png.')
    parser.add_argument('-d', '--device', type=str, help='Device to run inference on. Default to cpu.')
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    if output_path is None:
        output_path = input_path.split('.')[0] + '_out.png'
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
    model = checkpoint['model']
    model.eval()
    img = io.read_image(input_path) / 255
    img = img.unsqueeze(0)
    pred = model(img)
    out_img = (utils.argmax2img(pred[0].argmax(dim=0)) * 255).type(torch.uint8)
    io.write_png(out_img, output_path)

if __name__ == '__main__':
    main()