from math import log10
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from progress_bar import progress_bar
from util import niqe_metric


# Trainer super-class that the individual model trainers inherit from
class Trainer(object):
    def __init__(self, config, training_loader, valid_loader, model_type, diff=False):
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.valid_loader = valid_loader
        self.model_type = model_type
        self.diff = diff
        self.out_path = os.path.join("results/models/", self.model_type + ("_diff" if diff else ""))
        os.makedirs(self.out_path, exist_ok=True)

    def build_model(self):
        pass

    def save(self):
        model_out_path = os.path.join(self.out_path, "model.pth")
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.3f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def valid(self):
        self.model.eval()
        avg_psnr = 0
        avg_niqe = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                #calculate psnr
                mse = torch.mean(((prediction - target) ** 2), dim=[1, 2, 3])
                psnr = -10 * mse.log10().mean().item()
                avg_psnr += psnr
                #calculate niqe
                niqe = np.mean(niqe_metric.niqe(prediction.permute(0, 2, 3, 1).cpu().numpy() * 255, RGB=True, video_params=False))
                avg_niqe += niqe
                progress_bar(batch_num, len(self.valid_loader), 'PSNR: %.3f || NIQE: %.3f' % (avg_psnr / (batch_num + 1), avg_niqe / (batch_num + 1)))

        print("    Average PSNR: {:.3f} dB".format(avg_psnr / len(self.valid_loader)))

        return avg_psnr

    def run(self):
        best_psnr = 0
        self.build_model()
        n_params = sum(map(lambda x: x.numel(), self.model.parameters()))
        print("    {} params".format(n_params))
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            valid_psnr = self.valid()
            self.scheduler.step(epoch)
            if valid_psnr > best_psnr:
                self.save()
                best_psnr = valid_psnr

        print("    Best Average PSNR: {:.3f} dB".format(best_psnr))
        print("    {} params".format(n_params))


