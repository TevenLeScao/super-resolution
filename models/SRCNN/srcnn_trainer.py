import torch
import torch.backends.cudnn as cudnn

from trainer import Trainer
from models.SRCNN.model import Net


class SRCNNTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader):
        super(SRCNNTrainer, self).__init__(config, training_loader, valid_loader, "srcnn")

    def build_model(self):
        self.model = Net(num_channels=3, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)
