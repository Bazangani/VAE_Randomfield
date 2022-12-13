import gc
import warnings
from sklearn.manifold import TSNE
from plotly import express
import pandas
import numpy as np
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
batch_size = 50
hiden_size = 100
torch.cuda.is_available()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)


critertion = torch.nn.BCELoss(reduction='sum')

def load_ckp_VAE(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['vae_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizerd_state_dict'])

    return model, optimizer, checkpoint['epoch']

vae = VAE().to(device)
optim = torch.optim.Adam(params= vae.parameters(), lr=0.0001)
ckp_path_vae = "/home/bazanganif/Desktop/PhD/VAE/vae.pth"
vae, optim, start_epoch = load_ckp_VAE(ckp_path_vae, vae, optim)
vae.eval()

with torch.no_grad():
    latent = torch.randn((batch_size,hiden_size),device=device)

    gen_image = vae.decoder(latent)
    #print(gen_image.shape)
    gen_image =gen_image.to("cpu")

    #grid_image = torchvision.utils.make_grid(gen_image,nrow=10)
    #plt.imshow(grid_image.permute(1, 2, 0),cmap='hot')
    #plt.grid(False)
    #plt.show()
    print("TSNE starts")
    tsne =TSNE(n_components=2)
    tsne_result = tsne.fit_transform(gen_image.flatten().reshape(-1,1))
    print(tsne_result.shape)
    #fig = express.scatter(tsne_result, x=0, y=1)
    #fig.show()






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


for batch_idx, (image_batch,_) in enumerate(data_loader):
    inputs = image_batch.flatten().numpy()
    tsne_result_originial = tsne.fit_transform(inputs.reshape(-1, 1))
    color_original = np.zeros(120050)
    color_fake = np.ones(120050)
    color = np.concatenate([color_fake,color_original])
    print(tsne_result_originial.shape)
    tsne_result = np.concatenate([tsne_result,tsne_result_originial])

    fig = express.scatter(tsne_result, x=0, y=1,color=color)
    fig.show()
