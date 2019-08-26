import argparse
from math import log10

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

from models.DBPN.dbpn_trainer import DBPNTrainer
from models.DRCN.drcn_trainer import DRCNTrainer
from models.EDSR.edsr_trainer import EDSRTrainer
from models.FSRCNN.fsrcnn_trainer import FSRCNNTrainer
from models.SRCNN.srcnn_trainer import SRCNNTrainer
from models.SRGAN.srgan_trainer import SRGANTrainer
from models.SubPixelCNN.sub_pixel_trainer import SubPixelTrainer
from models.VDSR.vdsr_trainer import VDSRTrainer
from dataset.data import get_test_set, get_valid_set

from progress_bar import progress_bar

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='batch size')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default='test', help='use valid or test dataset. Default test')
parser.add_argument('--diff', default=False, action='store_true', help='is model differential ?')
parser.add_argument('--no-diff', dest='diff', action='store_false', help='is model differential ?')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srgan', help='choose which model is going to use')

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & valid dataset
    # ===========================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('===> Loading datasets')
    if args.dataset == 'test':
        test_set = get_test_set(args.upscale_factor)
    elif args.dataset == 'valid':
        test_set = get_valid_set(args.upscale_factor)
    else:
        raise NotImplementedError
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.batchSize, shuffle=False)

    file_name = args.model + "_generator.pth" if "gan" in args.model else "model.pth"
    model_name = args.model + ("_diff" if args.diff else "")
    model_path = "/home/teven/canvas/python/super-resolution/results/models/{}/{}".format(model_name, file_name)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model.to(device)
    model.eval()

    avg_psnr = 0
    avg_baseline_psnr = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            progress_bar(batch_num, len(test_data_loader), 'PSNR: %.3f' % (avg_psnr / (batch_num + 1)))

            baseline = F.interpolate(data, scale_factor=args.upscale_factor, mode='bilinear', align_corners=False)
            baseline_mse = criterion(baseline, target)
            baseline_psnr = 10 * log10(1 / baseline_mse.item())
            avg_baseline_psnr += baseline_psnr
            progress_bar(batch_num, len(test_data_loader), 'PSNR: %.3f' % (avg_baseline_psnr / (batch_num + 1)))

    print("    Average PSNR: {:.3f} dB".format(avg_psnr / len(test_data_loader)))
    print("    Average Baseline PSNR: {:.3f} dB".format(avg_baseline_psnr / len(test_data_loader)))


if __name__ == '__main__':
    main()
