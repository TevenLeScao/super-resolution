import torch
import math
import torch.nn as nn


class DENN(nn.Module):
    def __init__(self, num_channels, base_channels, feat_channels, num_stages, scale_factor):
        super(DENN, self).__init__()
