import torch.nn as nn
import torch.nn.functional as F

import torch
from torchsummary import summary
hidden_size = 400


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.FC = nn.Linear(in_features=hidden_size+200, out_features=1024*13*13)

        self.ConvT1 = nn.ConvTranspose2d(1024, 512, 5, 2, bias=False)
        self.ConvT2 = nn.ConvTranspose2d(512, 256, 5, 1, bias=False)
        self.ConvT3 = nn.ConvTranspose2d(256, 128, 5, 1, bias=False)
        self.ConvT4 = nn.ConvTranspose2d(128, 64, 5, 1, bias=False)
        self.ConvT5 = nn.ConvTranspose2d(64, 32, 5, 1, bias=False)
        self.ConvT6 = nn.ConvTranspose2d(32, 1, 5, 1, bias=False)


    def forward(self, input):

        yin = self.FC(input)
        y = yin.view(yin.size(0), 1024, 13, 13)

        y1 = self.ConvT1(y)
        y1_act = nn.LeakyReLU()(y1)


        y2 = self.ConvT2(y1)
        y2_act = nn.LeakyReLU()(y2)


        y3 = self.ConvT3(y2)
        y3_act = nn.LeakyReLU()(y3)

        y4 = self.ConvT4(y3)
        y4_act = nn.LeakyReLU()(y4)

        y5 = self.ConvT5(y4)
        y5_act = nn.LeakyReLU()(y5)

        y6 = self.ConvT6(y5)
        y6_act = nn.Sigmoid()(y6)
        return y6_act

