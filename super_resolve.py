import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default='/home/teven/canvas/python/super-resolution/dataset/urban100/images/test/img_049.png', help='input image to use')
parser.add_argument('--model', '-m', type=str, default='srgan', help='choose which model is going to use')
parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
parser.add_argument('--baseline', type=str, default='baseline.jpg', help='where to save the baseline image')
parser.add_argument('--diff', default=False, action='store_true', help='is model differential ?')
parser.add_argument('--no-diff', dest='diff', action='store_false', help='is model differential ?')
args = parser.parse_args()
print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
img = Image.open(args.input)
width, height = img.size
img.thumbnail((width / 4, height / 4), Image.ANTIALIAS)
baseline = img.resize((width, height))


# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
model_name = args.model
folder_name = args.model + ("_diff" if args.diff else "")
file_name = model_name + "_generator.pth" if "gan" in args.model else "model.pth"
model_path = "/home/teven/canvas/python/super-resolution/results/models/{}/{}".format(folder_name, file_name)
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)
data = (ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
data = data.to(device)

if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================
out = model(data)
out = out.cpu()
out_img = out.data[0].numpy()
out_img *= 255.0
out_img = out_img.clip(0, 255)
out_img = np.transpose(out_img, axes=(1, 2, 0))
print(out_img.shape)
out_img = Image.fromarray(np.uint8(out_img), 'RGB')

out_img.save(args.output)
print('output image saved to ', args.output)
baseline.save(args.baseline)
print('baseline image saved to ', args.baseline)
