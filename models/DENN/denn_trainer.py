import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from trainer import Trainer
from models.DENN.model import DENN


class DBPNTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader):
        super(DBPNTrainer, self).__init__(config, training_loader, valid_loader, "dbpn")

    def build_model(self):
        self.model = DENN(num_channels=3, base_channels=64, scale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init()
        self.criterion = nn.L1Loss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
