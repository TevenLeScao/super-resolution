import argparse

from torch.utils.data import DataLoader

from models.DBPN.dbpn_trainer import DBPNTrainer
from models.DRCN.drcn_trainer import DRCNTrainer
from models.EDSR.edsr_trainer import EDSRTrainer
from models.FSRCNN.fsrcnn_trainer import FSRCNNTrainer
from models.SRCNN.srcnn_trainer import SRCNNTrainer
from models.SRGAN.srgan_trainer import SRGANTrainer
from models.SubPixelCNN.sub_pixel_trainer import SubPixelTrainer
from models.VDSR.vdsr_trainer import VDSRTrainer
from dataset.data import get_training_set, get_valid_set

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='train batch size')
parser.add_argument('--validBatchSize', type=int, default=1, help='valid batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for. Default=30')
parser.add_argument('--lr', type=float, default=0.005, help='Learning Rate. Default=0.005')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srgan', help='choose which model is going to use')
parser.add_argument('--diff', default=False, action='store_true', help='is model differential ?')
parser.add_argument('--no-diff', dest='diff', action='store_false', help='is model differential ?')

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & valid dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set(args.upscale_factor)
    valid_set = get_valid_set(args.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_set, batch_size=args.validBatchSize, shuffle=False)

    if args.model == 'sub':
        model = SubPixelTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'edsr':
        model = EDSRTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'drcn':
        model = DRCNTrainer(args, training_data_loader, valid_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, valid_data_loader, diff=args.diff)
    elif args.model == 'dbpn':
        model = DBPNTrainer(args, training_data_loader, valid_data_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
