import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
hidden_size = 400


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.Conv1 = nn.Conv2d(2, 128, 3, 1, 1, bias=False)
        self.Conv2 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.Conv3 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.Conv4 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)


        # Mean and Var
        self.mean = nn.Linear( 1024*13*13,hidden_size)
        self.var = nn.Linear(1024*13*13,hidden_size)

    def sampling(self, mu, logVar):
            std = torch.exp(0.5 * logVar)
            esp = torch.randn_like(std)
            sample = mu + (esp * std)
            return sample

    def forward(self, input):
        x1 = self.Conv1(input)
        x1_act = nn.LeakyReLU()(x1)

        x2 = self.Conv2(x1_act)
        x2_act = nn.LeakyReLU()(x2)

        x3 = self.Conv3(x2_act)
        x3_act = nn.LeakyReLU()(x3)

        x4 = self.Conv4(x3_act)
        x4_act = nn.LeakyReLU()(x4)
        print(x4_act.shape)

        x5 = x4_act.view(x4_act.size(0),-1)

        latent_mu =self.mean(x5)
        latent_logVar = self.var(x5)
        z = self.sampling(latent_mu, latent_logVar)

        return latent_mu , latent_logVar,z

