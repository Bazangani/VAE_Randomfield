from torchsummary import summary
import torch.nn  as nn
import torch
hidden_size = 300
def conv(input_chanel,output_chanel):
    return nn.Conv2d(input_chanel,output_chanel,kernel_size=(5, 5),stride=2,padding=0)


# class U-Net

def dual_conv(input_channel,output_channel):
    conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(output_channel),
        nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(output_channel),
    )
    return conv

def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


def out_conv_gen(input_channel,output_channel):
    out_conv = nn.Sequential(
        nn.Conv2d(input_channel, output_channel,kernel_size=2,stride=1,padding=1),
        nn.Sigmoid()
    )
    return out_conv





class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()

        self.dwn_conv1 = dual_conv(1, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d (kernel_size=2,stride =2)

        self.mean = nn.Linear(1024 * 3 * 3, hidden_size)
        self.var = nn.Linear(1024 * 3 * 3, hidden_size)
        self.FC = nn.Linear(in_features=hidden_size , out_features=1024 * 3 * 3)


        # right side
        self.trans1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up_conv1 = dual_conv(1024,512)
        self.trans2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up_conv2 = dual_conv(512,256)
        self.trans3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride =2)
        self.up_conv3 = dual_conv(256,128)
        self.trans4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up_conv4 =dual_conv(128,64)
        self.out = out_conv_gen(64,1)

    def sampling(self, mu, logVar):
            std = torch.exp(0.5 * logVar)
            esp = torch.randn_like(std)
            sample = mu + (esp * std)
            return sample

    def forward(self, image):
            # left side
            x1 = self.dwn_conv1(image) #dual cov 3x3x3 witout padding and stride
            #print('shape of x1',x1.shape)
            x2 = self.maxpool(x1)  # Maxpooling with kernel 2x2x2 and stride 2
            #print('shape of x2', x2.shape)
            x3 = self.dwn_conv2(x2)
            #print('shape of x3', x3.shape)
            x4 = self.maxpool(x3)
            #print('shape of x4', x4.shape)
            x5 = self.dwn_conv3(x4)
            #print('shape of x5', x5.shape)
            x6 = self.maxpool(x5)
            #print('shape of x6', x6.shape)
            x7 = self.dwn_conv4(x6)
            #print('shape of x7', x7.shape)
            x8 = self.maxpool(x7)
            #print('shape of x8', x8.shape)
            x9 = self.dwn_conv5(x8)
            print('shape of x9', x9.shape)
            # right side

            view = x9.view(x9.size(0), -1)

            latent_mu = self.mean(view)
            latent_logVar = self.var(view)
            z = self.sampling(latent_mu, latent_logVar)
            flatt_f = self.FC(z)
            left_side = flatt_f.view(flatt_f.size(0), 1024, 3, 3)

        # forward pass for Right side


            x = self.trans1(left_side)
            #print('shape of x10', x.shape)
            y = crop_tensor(x, x7)
            #print('shape of x11', y.shape)
            #print(torch.cat([x, y], 1).shape)
            x = self.up_conv1(torch.cat([x, y], 1))
            #print('x12', x.shape)

            x = self.trans2(x)
            #print('shape of x13', x.shape)
            y = crop_tensor(x, x5)
            #print('shape of crop tensor', y.shape)
            x = self.up_conv2(torch.cat([x, y], 1))
            #print('shape of x14', x.shape)

            x = self.trans3(x)
            #print('shape of x15', x.shape)
            y = crop_tensor(x, x3)
            #print('crop_tensor y', y.shape)
            x = self.up_conv3(torch.cat([x, y], 1))
            #print('shape of x', x.shape)

            x = self.trans4(x)
            #print('shape of x', x.shape)
            y = crop_tensor(x, x1)
            #print('shape of y', y.shape)

            x = self.up_conv4(torch.cat([x, y[:, :, 0:48, 0:48]], 1))
            #print('shape of x', x.shape)

            x_recons = self.out(x)

            return x_recons,z,latent_mu,latent_logVar

def init_weight(m):
    if type(m) == nn.Conv3d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)
    if type(m) == nn.BatchNorm3d:
        nn.init.normal_(m.weight,mean=0.0,std=0.02)

def VAE():
    torch.cuda.is_available()
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    vae = Unet().to(device)
    vae.apply(init_weight)
    print(summary(vae, (1, 49, 49), batch_size=10))
    return vae

if __name__ == '__main__':
    batch_size = 1
    gen = VAE()