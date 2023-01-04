import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from collections import OrderedDict
import numpy as np


class MLPEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], im_dim=28**2):
        super().__init__()
        self.im_dim = im_dim
        layers = []
        indim = im_dim
        for dim in hidden_dim[:-1]:
            layers.append(nn.Linear(indim, dim))
            layers.append(nn.ELU())
            indim = dim
        layers.append(nn.Linear(indim, hidden_dim[-1]))
        self.fc1 = nn.Sequential(*layers)
        self.fc21 = nn.Linear(hidden_dim[-1], z_dim)
        self.fc22 = nn.Linear(hidden_dim[-1], z_dim)    

    def forward(self, x):
        hidden = F.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.clamp(F.softplus(self.fc22(hidden)), min=1e-3) 
        return z_loc, z_scale

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], im_dim=28**2):
        super().__init__()
        layers = []
        indim = z_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(indim, dim))
            layers.append(nn.ReLU())
            indim = dim
        layers.append(nn.Linear(indim, im_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.model(z))

fBase = 32

class MSEncoder(nn.Module):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, latent_dim, chan, hidden_dim = 512, prior_params=None):
        super(MSEncoder, self).__init__()
        self.prior_params = prior_params
        
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(chan, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(),
            # size: (fBase * 4) x 4 x 4
            nn.Conv2d(fBase * 4, hidden_dim, 4, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_dim)
        )
        self.c1 = nn.Linear(hidden_dim, latent_dim)
        self.c2 = nn.Linear(hidden_dim, latent_dim)
        self.c3 = nn.Linear(hidden_dim, latent_dim)
        self.c4 = nn.Linear(hidden_dim, latent_dim)

    # def inference(self, x):
    #     bs = x.shape[-4]
    #     x = x.view(-1, *x.shape[-3:])
    #     feat = self.enc(x).squeeze()
    #     feat = feat.view(-1, bs, feat.shape[-1]).squeeze()
    #     scale = self.c2(feat)
    #     return self.c1(feat), torch.clamp(F.softplus(scale), min=1e-4)

    # def forward(self, x):
    #     return self.inference(x)

    def forward(self, x):
        feat = self.enc(x).view(x.shape[0], -1)
        scale = self.c2(feat)
        params = self.c1(feat), torch.clamp(F.softplus(scale), min=1e-4)
        return params

class MSDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, latent_dim, chan, add_spectral_norm=False):
        super(MSDecoder, self).__init__()
        self.chan = chan
        if add_spectral_norm:
            self.dec = nn.Sequential(
                nn.utils.spectral_norm(nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True)),
                #nn.BatchNorm2d(fBase * 4),
                nn.ReLU(True),
                # size: (fBase * 4) x 4 x 4
                nn.utils.spectral_norm(nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True)),
                #nn.BatchNorm2d(fBase * 2),
                nn.ReLU(True),
                # size: (fBase * 2) x 8 x 8
                nn.utils.spectral_norm(nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True)),
                #nn.BatchNorm2d(fBase),
                nn.ReLU(True),
                # size: (fBase) x 16 x 16
                nn.utils.spectral_norm(nn.ConvTranspose2d(fBase, chan, 4, 2, 1, bias=True)),
                # Output size: 3 x 32 x 32
            )
        else:
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, fBase * 4, 4, 1, 0, bias=True),
                #nn.BatchNorm2d(fBase * 4),
                nn.ReLU(True),
                # size: (fBase * 4) x 4 x 4
                nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
                #nn.BatchNorm2d(fBase * 2),
                nn.ReLU(True),
                # size: (fBase * 2) x 8 x 8
                nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
                #nn.BatchNorm2d(fBase),
                nn.ReLU(True),
                # size: (fBase) x 16 x 16
                nn.ConvTranspose2d(fBase, chan, 4, 2, 1, bias=True),
                # Output size: 3 x 32 x 32
            )

    def forward(self, z):
        bs = z.shape[-2]
        z = z.unsqueeze(-1).unsqueeze(-1)
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(-1, bs, *out.size()[-3:]).squeeze(0)
        loc = torch.sigmoid(out)
        return loc
