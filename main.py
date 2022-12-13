import gc
import warnings
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
from Model.arch import VAE
import matplotlib.pyplot as plt
import os
import wandb
# dataloader
import sys
plt.style.use('ggplot')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if not sys.warnoptions:
    warnings.simplefilter("ignore")
wandb.init()
torch.cuda.empty_cache()
epochs = 2000
batch_size =10
torch.cuda.is_available()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


dataset = torchvision.datasets.ImageFolder('/home/bazanganif/Desktop/PhD/GAN_VAE/DATA',
                                           transforms.Compose([
                                               transforms.Grayscale(num_output_channels=1),
                                               #transforms.RandomCrop(20),
                                               transforms.Resize((49, 49)),

                                               transforms.ToTensor(),
                                               #transforms.Normalize(mean=[0.1],std=[0.9]),

                                           ]))
print(len(dataset), " images loaded")

data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          )

critertion = torch.nn.BCELoss(reduction='sum')

def Lower_band_loss(bce_loss, mu, logVar):


     KL_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2)-logVar.exp())
     return bce_loss + KL_divergence


def load_ckp_VAE(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['vae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizerd_state_dict'])

    return model, optimizer, checkpoint['epoch']

vae = VAE().to(device)
optim = torch.optim.Adam(params= vae.parameters(), lr=0.0001,weight_decay=1e-5)
ckp_path_vae = "/home/bazanganif/Desktop/PhD/VAE/vae.pth"
vae, optim, start_epoch = load_ckp_VAE(ckp_path_vae, vae, optim)

Loss =[]
fig, ax = plt.subplots(ncols=2)
fig_loss,ax_loss = plt.subplots(ncols=1)
for epoch in range(epochs):

    torch.cuda.empty_cache()
    gc.collect()

    VAE_loss = 0.0
    batch_idx = 0

    torch.backends.cudnn.benchmark = True

    for batch_idx, (image_batch,_) in enumerate(data_loader):
        if batch_idx + 1 == len(data_loader):
            break

        inputs = image_batch.to(device).detach()
        vae.zero_grad()

        recons_image, latent_mu, latent_logvar = vae(inputs)
        bce_loss = critertion(recons_image,inputs)

        LBL_loss = Lower_band_loss(bce_loss, latent_mu, latent_logvar)

        wandb.log({"Loss_re": LBL_loss})

        LBL_loss.backward()
        optim.step()

        print('Epoch [%d / %d] loss  : %f' % (epoch+1, epochs, LBL_loss))
        plt.close()

        ax[0].imshow(inputs[0,0,:,:].detach().cpu().numpy())
        ax[1].imshow(recons_image[0, 0, :, :].detach().cpu().numpy())
        fig.savefig('sample.png')
        wandb.log({'sample': wandb.Image(fig)})
        plt.close()

        #torch.save({
            #'epoch': epoch,
            #'vae_state_dict': vae.state_dict(),
           # 'optimizerd_state_dict': optim.state_dict(),
            #'loss_gen':LBL_loss }, "vae.pth")




