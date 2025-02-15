from math import log10
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg16
import numpy as np

from trainer import Trainer
# from models.SRGAN.model import Generator, Discriminator, DiffGenerator
from models.SRGAN.model import Generator, Discriminator, AugmentedDiffGenerator as DiffGenerator
from progress_bar import progress_bar
from util import niqe_metric


class SRGANTrainer(Trainer):
    def __init__(self, config, training_loader, valid_loader, diff):
        super(SRGANTrainer, self).__init__(config, training_loader, valid_loader, "srgan", diff)

        # SRGAN setup
        self.epoch_pretrain = 10
        self.feature_extractor = None
        self.num_residuals = 1
        self.netG = None
        self.netD = None
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None

    def build_model(self):
        if self.diff:
            self.netG = DiffGenerator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=16,
                                  num_channels=3).to(self.device)
        else:
            self.netG = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64,
                                  num_channels=3).to(self.device)
            self.netG.weight_init(mean=0.0, std=0.2)
        self.netD = Discriminator(base_filter=64, num_channels=3).to(self.device)
        self.feature_extractor = vgg16(pretrained=True)
        self.netD.weight_init(mean=0.0, std=0.2)
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            self.feature_extractor.cuda()
            cudnn.benchmark = True
            self.criterionG.cuda()
            self.criterionD.cuda()

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizerD = optim.SGD(self.netD.parameters(), lr=self.lr / 100, momentum=0.9, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 75, 100],
                                                        gamma=0.5)  # lr decay
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 75, 100],
                                                        gamma=0.5)  # lr decay

    @staticmethod
    def to_data(x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def save(self):
        g_model_out_path = os.path.join(self.out_path, "srgan_generator.pth")
        d_model_out_path = os.path.join(self.out_path, "srgan_discriminator.pth")
        torch.save(self.netG, g_model_out_path)
        torch.save(self.netD, d_model_out_path)
        print("Checkpoint saved to {}".format(g_model_out_path))
        print("Checkpoint saved to {}".format(d_model_out_path))

    def pretrain(self):
        self.netG.train()
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.netG.zero_grad()
            loss = self.criterionG(self.netG(data), target)
            loss.backward()
            self.optimizerG.step()

    def train(self):
        # models setup
        self.netG.train()
        self.netD.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # setup noise
            real_label = torch.ones(data.size(0), data.size(1)).to(self.device)
            fake_label = torch.zeros(data.size(0), data.size(1)).to(self.device)
            data, target = data.to(self.device), target.to(self.device)

            # Train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.netD(target)
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = self.netD(self.netG(data))
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.netG(data)
            g_fake = self.netD(g_real)
            gan_loss = self.criterionD(g_fake, real_label)
            mse_loss = self.criterionG(g_real, target)

            g_total = mse_loss + 1e-3 * gan_loss
            g_train_loss += g_total.item()
            g_total.backward()
            self.optimizerG.step()

            progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (
            g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))

        print("    Average G_Loss: {:.4f}".format(g_train_loss / len(self.training_loader)))

    def valid(self):
        self.netG.eval()
        avg_psnr = 0
        avg_niqe = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.netG(data)
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
        g_params = sum(map(lambda x: x.numel(), self.netG.parameters()))
        d_params = sum(map(lambda x: x.numel(), self.netD.parameters()))
        print("    {} G params".format(g_params))
        print("    {} D params".format(d_params))
        for epoch in range(1, self.epoch_pretrain + 1):
            self.pretrain()
            print("{}/{} pretrained".format(epoch, self.epoch_pretrain))

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            valid_psnr = self.valid()
            if valid_psnr > best_psnr:
                self.save()
                best_psnr = valid_psnr
            self.scheduler.step(epoch)

        print("    Best Average PSNR: {:.3f} dB".format(best_psnr))
        print("    {} G params".format(g_params))
        print("    {} D params".format(d_params))
