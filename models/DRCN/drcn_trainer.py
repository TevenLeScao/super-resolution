from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

from models.DRCN.model import Net
from progress_bar import progress_bar
from PIL import Image
from trainer import Trainer
from util import niqe_metric


class DRCNTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader):
        super(DRCNTrainer, self).__init__(config, training_loader, valid_loader, "drcn")

        # DRCN setup
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.loss_alpha = 1.0
        self.loss_alpha_zero_epoch = 25
        self.loss_alpha_decay = self.loss_alpha / self.loss_alpha_zero_epoch
        self.loss_beta = 0.001
        self.num_recursions = 16

    def build_model(self):
        self.model = Net(num_channels=3, base_channel=64, num_recursions=self.num_recursions, device=self.device).to(self.device)
        self.model.weight_init()
        self.criterion = nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        # setup optimizer and scheduler
        param_groups = [{'params': list(self.model.parameters())}]
        param_groups += [{'params': [self.model.w]}]
        self.optimizer = optim.Adam(param_groups, lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    def img_preprocess(self, data, interpolation='bicubic'):
        if interpolation == 'bicubic':
            interpolation = Image.BICUBIC
        elif interpolation == 'bilinear':
            interpolation = Image.BILINEAR
        elif interpolation == 'nearest':
            interpolation = Image.NEAREST

        size = list(data.shape)

        if len(size) == 4:
            target_height = int(size[2] * self.upscale_factor)
            target_width = int(size[3] * self.upscale_factor)
            out_data = torch.FloatTensor(size[0], size[1], target_height, target_width)
            for i, img in enumerate(data):
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((target_width, target_height), interpolation=interpolation),
                                                transforms.ToTensor()])

                out_data[i, :, :, :] = transform(img)
            return out_data
        else:
            target_height = int(size[1] * self.upscale_factor)
            target_width = int(size[2] * self.upscale_factor)
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((target_width, target_height), interpolation=interpolation),
                                            transforms.ToTensor()])
            return transform(data)

    def train(self):
        """
        data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
        """
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data = self.img_preprocess(data)  # resize input image size
            data, target = data.to(self.device), target.to(self.device)
            target_d, output = self.model(data)

            # loss1
            loss_1 = 0
            for d in range(self.num_recursions):
                loss_1 += (self.criterion(target_d[d], target) / self.num_recursions)

            # loss2
            loss_2 = self.criterion(output, target)

            # regularization
            reg_term = 0
            for theta in self.model.parameters():
                reg_term += torch.mean(torch.sum(theta ** 2))

            # total loss
            loss = self.loss_alpha * loss_1 + (1 - self.loss_alpha) * loss_2 + self.loss_beta * reg_term
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def valid(self):
        """
        data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        """
        self.model.eval()
        avg_psnr = 0
        avg_niqe = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.valid_loader):
                data = self.img_preprocess(data)  # resize input image size
                data, target = data.to(self.device), target.to(self.device)
                _, prediction = self.model(data)
                #calculate psnr
                mse = torch.mean(((prediction - target) ** 2), dim=[1, 2, 3])
                psnr = -10 * mse.log10().mean().item()
                avg_psnr += psnr
                #calculate niqe
                niqe = np.mean(niqe_metric.niqe(prediction.permute(0, 2, 3, 1).cpu().numpy() * 255, RGB=True, video_params=False))
                avg_niqe += niqe
                progress_bar(batch_num, len(self.valid_loader), 'PSNR: %.3f || NIQE: %.3f' % (avg_psnr / (batch_num + 1), avg_niqe / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.valid_loader)))

        return avg_psnr

    def run(self):
        best_psnr = 0
        self.build_model()
        n_params = sum(map(lambda x: x.numel(), self.model.parameters()))
        print("    {} params".format(n_params))
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.loss_alpha = max(0.0, self.loss_alpha - self.loss_alpha_decay)
            self.train()
            valid_psnr = self.valid()
            if valid_psnr > best_psnr:
                self.save()
                best_psnr = valid_psnr
            self.scheduler.step(epoch)

        print("    Best Average PSNR: {:.3f} dB".format(best_psnr))
        print("    {} params".format(n_params))
