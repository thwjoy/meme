import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from collections import OrderedDict
import numpy as np

# class ResizeConv2d(nn.Module):
    
#     def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
#         x = self.conv(x)
#         return x

# class BasicBlockEnc(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = in_planes*stride

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         if stride == 1:
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = torch.relu(out)
#         return out

# class BasicBlockDec(nn.Module):

#     def __init__(self, in_planes, stride=1):
#         super().__init__()

#         planes = int(in_planes/stride)

#         self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_planes)
#         # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

#         if stride == 1:
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential()
#         else:
#             self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.shortcut = nn.Sequential(
#                 ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
#                 nn.BatchNorm2d(planes)
#             )

#     def forward(self, x):
#         out = torch.relu(self.bn2(self.conv2(x)))
#         out = self.bn1(self.conv1(out))
#         out += self.shortcut(x)
#         out = torch.relu(out)
#         return out

# class ResNet18Enc(nn.Module):

#     def __init__(self, z_dim, num_Blocks=[2,2,2,2], nc=3):
#         super().__init__()
#         self.in_planes = 64
#         self.z_dim = z_dim
#         self.enc = nn.Sequential(
#                 nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),
#                 self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1),
#                 self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2),
#                 self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2),
#                 self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
#         )
#         self.c1 = nn.Linear(512, z_dim)
#         self.c2 = nn.Linear(512, z_dim)
#         self.c3 = nn.Linear(512, z_dim)

#     def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in strides:
#             layers += [BasicBlockEnc(self.in_planes, stride)]
#             self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.enc(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         return x.view(x.size(0), -1)

#     def generative(self, x):
#         x = self.forward(x)
#         return self.c1(x), torch.clamp(F.softplus(self.c3(x)), min=1e-3)

#     def inference(self, x):
#         x = self.forward(x)
#         return self.c1(x), torch.clamp(F.softplus(self.c2(x)), min=1e-3)


# class ResNet18Dec(nn.Module):

#     def __init__(self, z_dim, num_Blocks=[2,2,2,2], nc=3):
#         super().__init__()
#         self.in_planes = 512

#         self.linear = nn.Linear(z_dim, 512)

#         self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
#         self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
#         self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
#         self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
#         self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

#     def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
#         strides = [stride] + [1]*(num_Blocks-1)
#         layers = []
#         for stride in reversed(strides):
#             layers += [BasicBlockDec(self.in_planes, stride)]
#         self.in_planes = planes
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         x = self.linear(z)
#         x = x.view(-1, 512, 1, 1)
#         x = F.interpolate(x, scale_factor=4)
#         x = self.layer4(x)
#         x = self.layer3(x)
#         x = self.layer2(x)
#         x = self.layer1(x)
#         x = torch.sigmoid(self.conv1(x))
#         x = x.view(*z.shape[:-1], 3, 64, 64)
#         return x

# class View(nn.Module):
#     def __init__(self, size):
#         super(View, self).__init__()
#         self.size = size

#     def forward(self, tensor):
#         return tensor.view(self.size)
    
# class CELEBAEncoder(nn.Module):
#     def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
#         super().__init__()
#         # setup the three linear transformations used
#         self.z_dim = z_dim
#         self.enc = nn.Sequential(
#             nn.Conv2d(3, 32, 4, 2, 1), 
#             nn.ReLU(True),
#             nn.Conv2d(32, 32, 4, 2, 1),  
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, 4, 2, 1), 
#             nn.ReLU(True),
#             nn.Conv2d(64, 128, 4, 2, 1), 
#             nn.ReLU(True),
#             nn.Conv2d(128, hidden_dim, 4, 1),
#             nn.ReLU(True),
#             View((-1, hidden_dim*1*1))
#         )

#         self.locs = nn.Linear(hidden_dim, z_dim)
#         self.scales = nn.Linear(hidden_dim, z_dim)
#         self.scales_gen = nn.Linear(hidden_dim, z_dim)

#     def inference(self, x):
#         hidden = self.enc(x)
#         return self.locs(hidden), torch.clamp(F.softplus(self.scales(hidden)), min=1e-3)

#     def generative(self, x):
#         hidden = self.enc(x)
#         return self.locs(hidden), torch.clamp(F.softplus(self.scales_gen(hidden)), min=1e-3)    
        
# class CELEBADecoder(nn.Module):
#     def __init__(self, z_dim, hidden_dim=256, *args, **kwargs):
#         super().__init__()
#         # setup the two linear transformations used
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, hidden_dim),  
#             View((-1, hidden_dim, 1, 1)),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(hidden_dim, 128, 4),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 3, 4, 2, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, z):
#         m = self.decoder(z)
#         return m


# class Diagonal(nn.Module):
#     def __init__(self, dim):
#         super(Diagonal, self).__init__()
#         self.dim = dim
#         self.weight = nn.Parameter(torch.ones(self.dim))
#         self.bias = nn.Parameter(torch.zeros(self.dim))

#     def forward(self, x):
#         return x * self.weight + self.bias

# class Classifier(nn.Module):
#     def __init__(self, dim):
#         super(Classifier, self).__init__()
#         self.dim = dim
#         self.diag = Diagonal(self.dim)

#     def forward(self, x):
#         return self.diag(x)

# # class CondPrior(nn.Module):
# #     def __init__(self, dim):
# #         super(CondPrior, self).__init__()
# #         self.dim = dim
# #         self.diag_loc_true = nn.Parameter(torch.zeros(self.dim))
# #         self.diag_loc_false = nn.Parameter(torch.zeros(self.dim))
# #         self.diag_scale_true = nn.Parameter(torch.ones(self.dim))
# #         self.diag_scale_false = nn.Parameter(torch.ones(self.dim))

# #     def forward(self, x):
# #         loc = x * self.diag_loc_true + (1 - x) * self.diag_loc_false
# #         scale = x * self.diag_scale_true + (1 - x) * self.diag_scale_false
# #         return loc, torch.clamp(F.softplus(scale), min=1e-3)

# class CondPrior(nn.Module):
#     def __init__(self, dim):
#         super(CondPrior, self).__init__()
#         self.dim = dim
#         self.loc = nn.Linear(10, dim)
#         self.scale = nn.Linear(10, dim)

#     def forward(self, x):
#         return self.loc(x), torch.clamp(F.softplus(self.scale(x)), min=1e-3)

class MLPEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], im_dim=28**2):
        super().__init__()
        self.im_dim = im_dim
        # setup the three linear transformations used
        layers = []
        indim = im_dim
        for dim in hidden_dim[:-1]:
            layers.append(nn.Linear(indim, dim))
            #layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ELU())
            indim = dim
        layers.append(nn.Linear(indim, hidden_dim[-1]))
        self.fc1 = nn.Sequential(*layers)
        self.fc21 = nn.Linear(hidden_dim[-1], z_dim)
        self.fc22 = nn.Linear(hidden_dim[-1], z_dim)    

    def forward(self, x):
        hidden = F.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = torch.clamp(F.softplus(self.fc22(hidden)), min=1e-3) # torch.clamp(F.softplus(self.fc22(hidden)), min=1e-3)
        return z_loc, z_scale

    def inference(self, x):
        return self.forward(x)

    def generative(self, x):
        return self.forward(x)

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim=[512, 512], im_dim=28**2):
        super().__init__()
        # setup the two linear transformations used
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

# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)

# def product_of_priors(loc_p, scale_p, loc_con_p, scale_con_p, eps=1e-6):
#     bs = loc_con_p.shape[0]
#     scale_p = torch.clamp(F.softplus(scale_p), min=1e-4)
#     loc = torch.cat([loc_p.expand(bs, -1).unsqueeze(0), loc_con_p.unsqueeze(0)], dim=0)
#     scale = torch.cat([scale_p.expand(bs, -1).unsqueeze(0), scale_con_p.unsqueeze(0)], dim=0)
#     var       = scale.pow(2)
#     # precision of i-th Gaussian expert at point x
#     T         = 1. / (var + eps)
#     pd_mu     = torch.sum(loc * T, dim=0) / torch.sum(T, dim=0)
#     pd_var    = 1. / torch.sum(T, dim=0)
#     pd_scale = pd_var.sqrt()
#     return pd_mu, pd_scale

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

    def inference(self, x):
        bs = x.shape[-4]
        x = x.view(-1, *x.shape[-3:])
        feat = self.enc(x).squeeze()
        feat = feat.view(-1, bs, feat.shape[-1]).squeeze()
        scale = self.c2(feat)
        return self.c1(feat), torch.clamp(F.softplus(scale), min=1e-4)
        # if self.prior_params is not None:
        #     params = product_of_priors(*self.prior_params, *params)
        # return params

    def forward(self, x):
        return self.inference(x)

    def generative(self, x):
        feat = self.enc(x).view(x.shape[0], -1)
        scale = self.c2(feat)
        params = self.c1(feat), torch.clamp(F.softplus(scale), min=1e-4)
        # if self.prior_params is not None:
        #     params = product_of_priors(*self.prior_params, *params)
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

# maxSentLen = 32 
# embeddingDim = 128
# fBase = 32
# vocabSize = 1489

# class TextEncoder(nn.Module):
#     """ Generate latent parameters for sentence data. """

#     def __init__(self, z_dim):
#         super(TextEncoder, self).__init__()
#         self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
#         self.enc = nn.Sequential(
#             # input size: 1 x 32 x 128
#             nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(fBase),
#             nn.ReLU(True),
#             # size: (fBase) x 16 x 64
#             nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(fBase * 2),
#             nn.ReLU(True),
#             # size: (fBase * 2) x 8 x 32
#             nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # # size: (fBase * 4) x 4 x 16
#             nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # size: (fBase * 8) x 4 x 8
#             nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # size: (fBase * 8) x 4 x 4
#         )
#         self.c1 = nn.Conv2d(fBase * 4, z_dim, 4, 1, 0, bias=False)
#         self.c2 = nn.Conv2d(fBase * 4, z_dim, 4, 1, 0, bias=False)
#         # self.c3 = nn.Conv2d(fBase * 4, z_dim, 4, 1, 0, bias=False)
#         # self.c4 = nn.Conv2d(fBase * 4, z_dim, 4, 1, 0, bias=False)
#         # c1, c2 size: z_dim x 1 x 1

#     def forward(self, x):
#         return self.inference(x)

#     def inference(self, x):
#         e = self.enc(self.embedding(x.long()).unsqueeze(1))
#         return self.c1(e).squeeze(), torch.clamp(F.softplus(self.c2(e).squeeze()), min=1e-3)

#     def generative(self, x):
#         e = self.enc(self.embedding(x.long()).unsqueeze(1))
#         return self.c1(e).squeeze(), torch.clamp(F.softplus(self.c2(e).squeeze()), min=1e-3)

# class TextDecoder(nn.Module):
#     """ Generate a sentence given a sample from the latent space. """

#     def __init__(self, z_dim):
#         super(TextDecoder, self).__init__()
#         self.dec = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, fBase * 4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # size: (fBase * 8) x 4 x 4
#             nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # size: (fBase * 8) x 4 x 8
#             nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
#             nn.BatchNorm2d(fBase * 4),
#             nn.ReLU(True),
#             # size: (fBase * 4) x 8 x 32
#             nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(fBase * 2),
#             nn.ReLU(True),
#             # size: (fBase * 2) x 16 x 64
#             nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(fBase),
#             nn.ReLU(True),
#             # size: (fBase) x 32 x 128
#             nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
#             nn.ReLU(True)
#             # Output size: 1 x 64 x 256
#         )
#         # inverts the 'embedding' module upto one-hotness
#         self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

#     def forward(self, z):
#         z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
#         out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)
#         return self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize)


# class MatchingNet(nn.Module):
#     """ Contrastive network to match pairs """

#     def __init__(self, net_a, net_b, feat_dim, final_dim):
#         self.net_a = net_a
#         self.net_b = net_b
#         self.final_dim = final_dim

#         self.net = nn.Sequential(
#             nn.Linear(feat_dim, final_dim),
#             nn.ReLU(),
#             nn.Linear(feat_dim, final_dim),
#         )

#     def forward(self, a, b):
#         feats_a, _ = self.net_a(a)
#         feats_b, _ = self.net_b(b)
#         bs = a.shape[0]
#         feats_a = feats_a.unsqueeze(1).expand(bs, *feats_a.shape)
#         feats_b = feats_b.unsqueeze(0).expand(bs, *feats_b.shape)

#         # compute dot product
#         cos = nn.CosineSimilarity(dim=0, eps=1e-6)
#         return nn.Softmax(cos(feats_a, feats_b), dim=-1)


# class LSTMEncoder(nn.Module):
#     """ Generate latent parameters for sentence data. """

#     def __init__(self, z_dim):
#         super(LSTMEncoder, self).__init__()
#         bidirectional = True
#         self.mult = 2 if bidirectional else 1
#         self.hidden_dim = 256
#         self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
#         self.dropout = nn.Dropout(0.2)
#         self.lstm = nn.LSTM(embeddingDim, self.hidden_dim, batch_first=True, bidirectional=bidirectional)
#         self.loc = nn.Linear(self.hidden_dim * self.mult, z_dim)
#         self.scale = nn.Linear(self.hidden_dim * self.mult, z_dim)
#         # c1, c2 size: z_dim x 1 x 1

#     def forward(self, x):
#         x = self.embedding(x.long())
#         x = self.dropout(x)
#         lstm_out, (ht, ct) = self.lstm(x)
#         ht = ht.view(x.shape[0], self.hidden_dim * self.mult)
#         return self.loc(ht), torch.clamp(F.softplus(self.scale(ht)), min=1e-3)

# class LSTMDecoder(nn.Module):
#     """ Generate a sentence given a sample from the latent space. """

#     def __init__(self, z_dim, drop_prob_start=0.5):
#         super(LSTMDecoder, self).__init__()
#         bidirectional = True
#         self.mult = 2 if bidirectional else 1
#         self.drop_prob = drop_prob_start
#         self.z_dim = z_dim
#         self.hidden_dim = 256
#         self.latent_to_hidden = nn.Linear(self.z_dim, self.hidden_dim * self.mult)
#         self.lstm = nn.LSTM(embeddingDim, self.hidden_dim,
#                             batch_first=True, bidirectional=bidirectional)
#         self.toVocabSize = nn.Linear(self.hidden_dim * self.mult, vocabSize)

#     def forward(self, z, s, embedding_fn, w2i):
#         hidden = self.latent_to_hidden(z).view(self.mult, z.shape[0], -1)
#         # do word dropout
#         s = s.clone()
#         prob = torch.rand_like(s)
#         for tok in ['<sos>', '<eos>', '<pad>']:
#             inds = torch.where(s.cpu() == w2i[tok])
#             prob[inds] = 1.0
#         inds_to_drop = torch.where(prob < self.drop_prob)
#         s[inds_to_drop] = w2i['<exc>']
#         embedding = embedding_fn(s.long())
#         output, (hn, cn) = self.lstm(embedding, (hidden, torch.zeros_like(hidden)))
#         preds = self.toVocabSize(output.reshape(-1, self.hidden_dim * self.mult)).view(-1, maxSentLen, vocabSize)
#         return preds

#     def sample_greedy(self, z, embedding_fn, w2i):
#         words = torch.zeros((z.shape[0], 1), device=z.device, dtype=torch.long).fill_(w2i['<sos>'])
#         this_word = words
#         with torch.no_grad():
#             state_h = self.latent_to_hidden(z).view(self.mult, z.shape[0], -1)
#             state_c = torch.zeros_like(state_h)
#             for i in range(0, maxSentLen):
#                 emb = embedding_fn(words)
#                 out, (state_h, state_c) = self.lstm(emb, (state_h, state_c))
#                 preds = self.toVocabSize(state_h.reshape(-1, self.hidden_dim * self.mult)).view(-1, 1, vocabSize)
#                 this_word = preds.argmax(dim=-1)
#                 words = torch.cat([words, this_word], dim=1)
#         return words

#     # def sample(self, z, embedding_fn, w2i):
#     #     words = torch.zeros((z.shape[0], 1), device=z.device, dtype=torch.long).fill_(w2i['<sos>'])
#     #     this_word = words
#     #     with torch.no_grad():
#     #         state_h = self.latent_to_hidden(z).view(self.mult, z.shape[0], -1)
#     #         state_c = torch.zeros_like(state_h)
#     #         for i in range(0, maxSentLen):
#     #             emb = embedding_fn(this_word)
#     #             out, (state_h, state_c) = self.lstm(emb, (state_h, state_c))
#     #             preds = self.toVocabSize(state_h.reshape(-1, self.hidden_dim * self.mult)).view(-1, 1, vocabSize)
#     #             this_word = dist.Categorical(logits=preds).sample()
#     #             words = torch.cat([words, this_word], dim=1)
#     #     return words

#     def sample(self, z, embedding_fn, w2i):
#         inp = torch.zeros((z.shape[0], maxSentLen), device=z.device, dtype=torch.long)
#         inp[:, 0].fill_(w2i['<sos>'])
#         inp[:, 1:].fill_(w2i['<exc>'])
#         with torch.no_grad():
#             emb = embedding_fn(inp)
#             state_h = self.latent_to_hidden(z).view(self.mult, z.shape[0], -1)
#             state_c = torch.zeros_like(state_h)
#             out, (state_h, state_c) = self.lstm(emb, (state_h, state_c))
#             preds = self.toVocabSize(out.reshape(-1, self.hidden_dim * self.mult)).view(-1, maxSentLen, vocabSize).argmax(dim=-1)
#         return preds



# # class CUBClassifier(nn.Module):
# #     def __init__(self, z_dim, attr_dims):
# #         super().__init__()
# #         self.z_dim = z_dim
# #         self.attr_dims = attr_dims
# #         hidden_dim = 128
# #         self.net = nn.Sequential(nn.Linear(self.z_dim, hidden_dim),
# #                                  nn.ELU(),
# #                                  nn.Linear(hidden_dim, hidden_dim),
# #                                  nn.ELU(),
# #                                  nn.Linear(hidden_dim, hidden_dim),
# #                                  nn.ELU())
        
# #         self.finals = []
# #         for key, v in attr_dims.items():
# #             self.finals.append(nn.Linear(hidden_dim, len(v)))

# #     def forward(self, z):
# #         pass

# class CUBClassifier(nn.Module):
#     def __init__(self, attr_dims):
#         super().__init__()
#         self.attr_dims = attr_dims
        
#         self.layers = OrderedDict()
#         for key, v in attr_dims.items():
#             self.layers[key] = nn.Linear(1, len(v)).cuda()

#     def forward(self, z):
#         preds = OrderedDict()
#         for i, (key, l) in enumerate(self.layers.items()):
#             preds[key] = l(z[:, i].unsqueeze(1))
#         return preds

# class LocScale(nn.Module):
#     def __init__(self, classes):
#         super().__init__()
#         self.classes = classes
#         self.loc = nn.Linear(classes, 1)
#         self.scale = nn.Linear(classes, 1)

#     def forward(self, x):
#         return self.loc(x), torch.clamp(F.softplus(self.scale(x)), min=1e-3)

# class CUBCondPrior(nn.Module):
#     def __init__(self, z_dim, u_dim, attr_dims):
#         super().__init__()
#         self.z_dim = z_dim
#         self.u_dim = u_dim
#         self.attr_dims = attr_dims
#         hidden_dim = 128
#         y_dim = hidden_dim - self.u_dim
#         self.text = MLPEncoder(self.z_dim, [hidden_dim] * 3, hidden_dim)
#         self.y_proj = nn.Linear(10 * len(attr_dims), y_dim)
#         self.layers = OrderedDict()
#         for key, v in attr_dims.items():
#             self.layers[key] = nn.Linear(len(v), 10).cuda()



#     def forward(self, y, u):
#         #from_y_loc, from_y_scale = [], []
#         y_feats = []
#         for key, l in self.layers.items():
#             y_feats.append(l(y[key]))
#             # loc, scale = l(y[key])
#             # from_y_loc.append(loc)
#             # from_y_scale.append(scale)
#         feats = torch.stack(y_feats, dim=1).view(u.shape[0], -1)
#         y_feats = self.y_proj(feats)
        
#         return self.text(torch.cat([y_feats, u], dim=1))
        

# class FeatsEnc(nn.Module):
#     """ Generate latent parameters for CUB image feature. """

#     def __init__(self, latent_dim, n_c=2048):
#         super(FeatsEnc, self).__init__()
#         dim_hidden = 256
#         self.enc = nn.Sequential()
#         for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
#             self.enc.add_module("layer" + str(i), nn.Sequential(
#                 nn.Linear(n_c // (2 ** i), n_c // (2 ** (i + 1))),
#                 nn.ELU(inplace=True),
#             ))
#         # relies on above terminating at dim_hidden
#         self.fc21 = nn.Linear(dim_hidden, latent_dim)
#         self.fc22 = nn.Linear(dim_hidden, latent_dim)
#         # self.fc21g = nn.Linear(dim_hidden, latent_dim)
#         # self.fc22g = nn.Linear(dim_hidden, latent_dim)

#     def forward(self, x):
#         e = self.enc(x)
#         return self.fc21(e), torch.clamp(F.softplus(self.fc22(e)), min=1e-4)

#     def inference(self, x):
#         e = self.enc(x)
#         return self.fc21(e), torch.clamp(F.softplus(self.fc22(e)), min=1e-4)

#     def generative(self, x):
#         e = self.enc(x)
#         return self.fc21(e), torch.clamp(F.softplus(self.fc22(e)), min=1e-4)

# class FeatsDec(nn.Module):
#     """ Generate a CUB image feature given a sample from the latent space. """

#     def __init__(self, latent_dim, n_c=2048):
#         super(FeatsDec, self).__init__()
#         self.n_c = n_c
#         dim_hidden = 256
#         self.dec = nn.Sequential()
#         for i in range(int(torch.tensor(n_c / dim_hidden).log2())):
#             indim = latent_dim if i == 0 else dim_hidden * i
#             outdim = dim_hidden if i == 0 else dim_hidden * (2 * i)
#             self.dec.add_module("out_t" if i == 0 else "layer" + str(i) + "_t", nn.Sequential(
#                 nn.Linear(indim, outdim),
#                 nn.ELU(inplace=True),
#             ))
#         # relies on above terminating at n_c // 2
#         self.fc31 = nn.Linear(n_c // 2, n_c)

#     def forward(self, z):
#         p = self.dec(z.view(-1, z.size(-1)))
#         mean = self.fc31(p).view(*z.size()[:-1], -1)
#         return mean#, torch.tensor([0.01]).to(mean.device)





