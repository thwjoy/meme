import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torch.distributions as dist
import os
from .networks import MLPDecoder, MLPEncoder, MSEncoder, MSDecoder


def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):

    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:                                                                                                                                                                                                         
        scale_p = torch.ones_like(scale_q)

    kl = 0.5 * (2 * scale_p.log() - 2 * scale_q.log() + \
                (locs_q - locs_p).pow(2) / scale_p.pow(2) + \
                scale_q.pow(2) / scale_p.pow(2) - torch.ones_like(locs_q))

    return kl.sum(dim=-1)

def compute_kl_enum(locs_q, scale_q, locs_p, scale_p,
                    z, logit_probs):
    # vamp prior
    log_qz = dist.Normal(locs_q, scale_q).log_prob(z).sum(dim=-1)
    locs_p = locs_p.unsqueeze(0).expand(locs_q.shape[0], -1, -1)
    scale_p = scale_p.unsqueeze(0).expand(locs_q.shape[0], -1, -1)

    mix = dist.Categorical(logits=logit_probs)
    comp = dist.Independent(dist.Normal(locs_p, scale_p), 1)
    gmm = dist.mixture_same_family.MixtureSameFamily(mix, comp)

    kl = log_qz - gmm.log_prob(z)

    return kl


class MEME_BASE(nn.Module):
    def __init__(self, z_dim, device, pseudo_samples_a, pseudo_samples_b):
        super(MEME_BASE, self).__init__()
        self.z_dim = z_dim
        self.device = device

        self.pseudo_samples_a = nn.Parameter(pseudo_samples_a)
        self.pseudo_samples_b = nn.Parameter(pseudo_samples_b)

    def get_mixture_prior_params(self, samples):
        return self.cond_prior.generative(samples)

    def img_image_likelihood(self, pred, targ):
        scale = torch.ones_like(pred) * self.scale_val
        return dist.Laplace(pred, scale).log_prob(targ).sum(dim=(-3, -2, -1))

    def img_vec_likelihood(self, pred, targ):
        scale = torch.ones_like(pred) * self.scale_val
        return dist.Laplace(pred, scale).log_prob(targ).sum(dim=(-1))

    def match(self, mode_a, mode_b, direction='bi'):
        return self.run(mode_a, mode_b, direction, self.run_match)

    def unsup(self, mode_a, mode_b, direction):
        return self.run(mode_a, mode_b, direction, self.run_unsup)

    def classifier_loss_img(self, data, targ, likelihood, k=10, z_sample=None):
        post_params = self.encoder.inference(data)
        z = dist.Normal(*post_params).rsample([k])
        preds = self.classifier(z).view(-1, *targ.shape)
        targ_ = targ.unsqueeze(0).expand(k, *targ.shape)
        probs = likelihood(preds, targ_)
        probs = probs.view(1, k, -1) # no_z x no_k x bs
        if z_sample is not None:
            bs = data.shape[0]
            z_sample = z_sample.view(-1, bs, self.z_dim)
            preds_samples = self.classifier(z_sample)
            targ = targ.unsqueeze(0).expand(-1, *targ.shape)
            probs_samples = likelihood(preds_samples, targ)
            probs = probs.expand(z_sample.shape[0], -1, -1)
            probs = torch.cat((probs, probs_samples.unsqueeze(1)), dim=1)
        log_qts = torch.logsumexp(probs, dim=1) - np.log(probs.shape[1])
        return log_qts, preds
    
    def run_match(self, data, targ, k=1):
        bs = data.shape[0]
        post_params = self.encoder.inference(data)
        z = dist.Normal(*post_params).rsample([k])
        pred  = self.classifier(z)
        log_qtz = self.likelihood_t(pred, targ)

        c_prior_params = self.cond_prior.generative(targ)

        kl = compute_kl(*post_params, *c_prior_params)

        recon = self.decoder(z)
        log_psz = self.likelihood_s(recon, data)

        log_qts, pred_samples = self.classifier_loss_img(data, targ, self.likelihood_t, k=10, z_sample=z)
        w = torch.exp(log_qtz - log_qts)

        loss = w.detach() * (log_psz - log_qtz - kl) + self.classifier_scale_sup * (log_qts)

        return -loss.mean()


    def run_unsup(self, data, *args):
        bs = data.shape[0]
        post_params = self.encoder.inference(data)
        z = dist.Normal(*post_params).rsample()

        prior_params = self.get_mixture_prior_params(self.pseudo_samples[:bs])

        kl = compute_kl_enum(*post_params, *prior_params,
                             z=z, logit_probs=torch.ones((bs, bs), device=data.device))

        log_psz = self.likelihood_s(self.decoder(z), data)

        loss = - (log_psz  - kl).mean()

        return loss.mean()

    def save_models(self, path='./data'):
        torch.save(self.b_to_z, os.path.join(path,'b_to_z.pt'))
        torch.save(self.a_to_z, os.path.join(path,'a_to_z.pt'))
        torch.save(self.z_to_b, os.path.join(path, 'z_to_b.pt'))
        torch.save(self.z_to_a, os.path.join(path, 'z_to_a.pt'))
        torch.save(self.pseudo_samples_b, os.path.join(path, 'pseudo_samples_b'))
        torch.save(self.pseudo_samples_a, os.path.join(path, 'pseudo_samples_a'))


class MEME_IMAGE_IMAGE(MEME_BASE):
    def __init__(self, z_dim, device, pseudo_samples_a, pseudo_samples_b):
        super(MEME_IMAGE_IMAGE, self).__init__(z_dim, device, 
                                              pseudo_samples_a,
                                              pseudo_samples_b)
        self.to(device)
        self.classifier_scale_sup = 10
        self.classifier_scale_unsup = 1

    def tsne_plot(self, loader, device, data_dir, epoch, direction):
        with torch.no_grad():
            enc_feats = []
            labs = []
            for i, (svhn, mnist, y) in enumerate(loader):
                labs.append(y)
                az = dist.Normal(*self.a_to_z.inference(svhn.to(device))).sample()
                enc_feats.append(az.cpu())
                labs.append(y + 10)
                bz = dist.Normal(*self.b_to_z.inference(mnist.to(device))).sample().cpu()
                enc_feats.append(bz)
                if i > 5:
                    break
            enc_feats = torch.cat(enc_feats, dim=0)
            labs = torch.cat(labs, dim=0)

            model_tsne_high = TSNE(n_components=2, random_state=0)
            z_embed = model_tsne_high.fit_transform(enc_feats)
            fig = plt.figure(figsize=(7, 5))
            for ic in range(10):
                ind_class = np.where(labs == ic)
                color = plt.cm.tab20(2*ic)
                plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color, label="s%i" % ic)
            for ic in range(10):
                ind_class = np.where(labs == ic + 10)
                color = plt.cm.tab20(2*ic+1)
                plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color, label="m%i" % ic)
            plt.title("Latent Variable T-SNE")
            plt.legend(loc='center right')
            name = 'embedding_%i_%s.png' % (epoch, direction)
            if data_dir is not None:
                fig.savefig(os.path.join(data_dir, name))
            
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            plt.close('all')
            ncols, nrows = fig.canvas.get_width_height()
            shape = (nrows, ncols, 3)
            fig = np.rollaxis(np.fromstring(buf, dtype=np.uint8).reshape(shape), 2, 0)

            return {'latent': fig}

    def svhn_to_mnist(self, data):
        z = dist.Normal(*self.a_to_z.inference(data)).sample()
        return self.z_to_b(z)

    def mnist_to_svhn(self, img):
        return self.z_to_a(dist.Normal(*self.b_to_z.inference(img)).sample())

    def svhn_to_svhn(self, img):
        return self.z_to_a(dist.Normal(*self.a_to_z.inference(img)).sample())

    def mnist_to_mnist(self, img):
        z = dist.Normal(*self.b_to_z.inference(img)).sample()
        return self.z_to_b(z)

class MEME_MNIST_SVHN(MEME_IMAGE_IMAGE):
    def __init__(self, z_dim, device,
                 pseudo_samples_a, pseudo_samples_b):
        super(MEME_MNIST_SVHN, self).__init__(z_dim, device,
                                              pseudo_samples_a,
                                              pseudo_samples_b)
        self.scale_val = 1.0
        self.b_to_z = MLPEncoder(self.z_dim)
        self.z_to_b = MLPDecoder(self.z_dim)
        self.a_to_z = MSEncoder(self.z_dim, 3)
        self.z_to_a = MSDecoder(self.z_dim, 3)

        self.to(device)

    def run(self, mode_a, mode_b, direction, fn):
        if direction == 'a2b':
            self.direction = 'a2b'
            self.encoder = self.a_to_z
            self.decoder = self.z_to_a
            self.cond_prior = self.b_to_z
            self.classifier = self.z_to_b
            self.likelihood_s = self.img_image_likelihood
            self.likelihood_t = self.img_vec_likelihood
            self.pseudo_samples = self.pseudo_samples_b
            data, targ = mode_a, mode_b
        elif direction == 'b2a':
            self.direction = 'b2a'
            self.encoder = self.b_to_z
            self.decoder = self.z_to_b
            self.cond_prior = self.a_to_z
            self.classifier = self.z_to_a
            self.likelihood_s = self.img_vec_likelihood
            self.likelihood_t = self.img_image_likelihood
            self.pseudo_samples = self.pseudo_samples_a
            data, targ = mode_b, mode_a
        elif direction == 'bi':
            loss_a2b = self.run(mode_a, mode_b, 'a2b', fn)
            loss_b2a = self.run(mode_a, mode_b, 'b2a', fn)
            return loss_b2a + loss_a2b
        elif direction == 'alt':
            if self.direction == 'b2a':
                return self.run(mode_a, mode_b, 'a2b', fn)
            elif self.direction == 'a2b':
                return self.run(mode_a, mode_b, 'b2a', fn)
        return fn(data, targ)
