import torch
import torch.backends.cudnn as cudnn

from trainer import Trainer
from models.SubPixelCNN.model import Net


class SubPixelTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader):
        super(SubPixelTrainer, self).__init__(config, training_loader, valid_loader, "sub_pixel")

    def build_model(self):
        self.model = Net(upscale_factor=self.upscale_factor).to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
