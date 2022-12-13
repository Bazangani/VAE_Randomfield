import torch
from Model.Encoder import Encoder
from Model.Decoder import Decoder
import torch.nn as nn
from torchsummary import summary
hidden_size = 200

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class VariationaAutoencoder(nn.Module):
    def __init__(self):
        super(VariationaAutoencoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
       # self.encoder = nn.Sequential(
           # nn.Conv2d(1, 64, 3, 1, 1, bias=False),
           # nn.LeakyReLU(),
           # nn.Conv2d(64, 128, 3, 2, 1, bias=False),
           # nn.LeakyReLU(),
            #nn.Conv2d(128, 256, 3, 2, 1, bias=False),
           # nn.LeakyReLU(),
            #nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            #nn.LeakyReLU(),

        #)



        # Mean and Var
        #self.mu = nn.Linear( 512*13*13,hidden_size)
        #self.var = nn.Linear(512*13*13,hidden_size)


        # DECODEDER
        #self.FC = nn.Linear(in_features=hidden_size, out_features=512*13*13)

        #self.ConvT1 = nn.ConvTranspose2d(512, 256, 5, 2, bias=False)
        #self.ConvT2 = nn.ConvTranspose2d(256, 128, 5, 1, bias=False)
        #self.ConvT3 = nn.ConvTranspose2d(128, 64, 5, 1, bias=False)
        #self.ConvT4 = nn.ConvTranspose2d(64, 32, 5, 1, bias=False)
        #self.ConvT5 = nn.ConvTranspose2d(32, 16, 5, 1, bias=False)
        #self.ConvT6 = nn.ConvTranspose2d(16, 1, 5, 1, bias=False)


    def forward(self,x,h):
        latent_mu, latent_logVar, z = self.encoder(x)
        h = h.repeat(1,200)
        z_h = torch.cat((z,h),dim=1)
        recons = self.decoder(z_h)


        #x1 = self.Conv1(x)
        #x1_act = nn.LeakyReLU()(x1)

        #x2 = self.Conv2(x1_act)
        #x2_act = nn.LeakyReLU()(x2)

#        x3 = self.Conv3(x2_act)
 #       x3_act = nn.LeakyReLU()(x3)
#      x4 = self.Conv4(x3_act)
 #       x4_act = nn.LeakyReLU()(x4)

  #      x5 = x4_act.view(x4_act.size(0), -1)

   #     latent_mu = self.mu(x5)
    #    latent_logVar = self.var(x5)

     #   latent_sample = self.sampling(latent_mu, latent_logVar)


      #  yin = self.FC(latent_sample)

       # y = yin.view(yin.size(0), 512, 13, 13)

        #y1 = self.ConvT1(y)
        #y1_act = nn.LeakyReLU()(y1)

        #y2 = self.ConvT2(y1_act)
        #y2_act = nn.LeakyReLU()(y2)

        #y3 = self.ConvT3(y2_act)
        #y3_act = nn.LeakyReLU()(y3)

        #y4 = self.ConvT4(y3_act)
        #y4_act = nn.LeakyReLU()(y4)

        #y5 = self.ConvT5(y4_act)
        #y5_act = nn.LeakyReLU()(y5)

        #y6 = self.ConvT6(y5_act)
        #x_recons = nn.Sigmoid()(y6)

        return recons, latent_mu, latent_logVar


def VAE():
    torch.cuda.is_available()

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    vae = VariationaAutoencoder().to(device)
    #vae.apply(weights_init)
    #summary(vae, (torch.randn(1, 49,49).shape), batch_size=2)
    return vae



if __name__ == '__main__':
    batch_size = 10
    print(batch_size)
    gen = VAE()
