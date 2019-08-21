import torch
import torch.backends.cudnn as cudnn

from models.FSRCNN.model import Net
from trainer import Trainer


class FSRCNNTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader):
        super(FSRCNNTrainer, self).__init__(config, training_loader, valid_loader, "fsrcnn")

    def build_model(self):
        self.model = Net(num_channels=3, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
