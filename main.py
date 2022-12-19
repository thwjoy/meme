import argparse
from logging import disable
import shutil
from random import *
import torch
from torch.nn import parameter
import torchvision
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import torch.autograd
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils.dataset_cached import setup_data_loaders, MNIST_SVHN
from models.meme import MEME_MNIST_SVHN

import numpy as np
import os


def set_seed(seed):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):
    data_shape = (3, 32, 32)
    mnist_shape = (1, 28, 28)

    data_loaders = setup_data_loaders(args.batch_size,
                                      sup_frac=args.sup_frac,
                                      root='./data/datasets')

    if args.sup_frac != 1.0:
        pseudo_samples_a, pseudo_samples_b, _ = next(iter(data_loaders['unsup']))
    else:
        pseudo_samples_a, pseudo_samples_b, _ = next(iter(data_loaders['sup']))

    device = torch.device("cuda:0" if args.cuda else "cpu")

    vae = MEME_MNIST_SVHN(z_dim=args.z_dim,
                         device=device,
                         pseudo_samples_a=pseudo_samples_a.to(device),
                         pseudo_samples_b=pseudo_samples_b.to(device))

    optim = torch.optim.Adam(params=vae.parameters(), lr=args.learning_rate)

    it = 0

    # run inference for a certain number of epochs
    for epoch in range(0, args.num_epochs):
        # # # compute number of batches for an epoch
        if args.sup_frac == 1.0: # fully supervised
            batches_per_epoch = len(data_loaders["sup"])
            period_sup_batches = 1
            sup_batches = batches_per_epoch
        elif args.sup_frac > 0.0: # semi-supervised
            sup_batches = len(data_loaders["sup"])
            unsup_batches = len(data_loaders["unsup"])
            batches_per_epoch = sup_batches + unsup_batches
            period_sup_batches = int(batches_per_epoch / sup_batches)
        else:
            assert False, "Data frac not correct"

        epoch_losses_sup = 0.0
        epoch_losses_unsup = 0.0
        epoch_w = 0.0

        # setup the iterators for training data loaders
        if args.sup_frac != 0.0:
            sup_iter = iter(data_loaders["sup"])
        if args.sup_frac != 1.0:
            unsup_iter = iter(data_loaders["unsup"])

        # count the number of supervised batches seen in this epoch
        ctr_sup = 0
        num_sups = 0
        num_unsups = 0

        for i in tqdm(range(batches_per_epoch)):
            it += 1
            # whether this batch is supervised or not
            is_supervised = (i % period_sup_batches == 0) and ctr_sup < sup_batches

            # extract the corresponding batch
            if is_supervised:
                data = next(sup_iter)
                ctr_sup += 1
            else:
                data = next(unsup_iter)

            mode_a = data[0].to(device)
            mode_b = data[1].to(device)

            if is_supervised: 
                num_sups += 1

                loss = vae.match(mode_a=mode_a, mode_b=mode_b)
                loss.backward()
                epoch_losses_sup += loss.detach().item()
   
            else:
                num_unsups += 1

                if args.missing is None:
                    loss = vae.unmatch(mode_a=mode_a, mode_b=mode_b, direction=args.direction)
                elif args.missing == 'b':
                    loss = vae.unsup(mode_a=mode_a, mode_b=None, direction='a2b')
                elif args.missing == 'a':
                    loss = vae.unsup(mode_a=None, mode_b=mode_b, direction='b2a')
                elif args.missing == 'ab':
                    r = random()
                    if r < 0.5:
                        loss = vae.unsup(mode_a=None, mode_b=mode_b, direction='b2a')
                    else:
                        loss = vae.unsup(mode_a=mode_a, mode_b=None, direction='a2b')

                epoch_losses_unsup += loss.detach().item()
                loss.backward()
 
            optim.step()
            optim.zero_grad()            
                      
        if epoch % 10 == 0:
            with torch.no_grad():
                mnist, svhn = MNIST_SVHN.fixed_imgs
                mnist = mnist.to(device)
                svhn = svhn.to(device)
                recon_mnist = F.pad(vae.mnist_to_mnist(mnist).view(-1, *mnist_shape), (2, 2, 2, 2),
                                    mode='constant', value=0)
                recon_svhn = vae.svhn_to_svhn(svhn)
                svhn_to_mnist = F.pad(vae.svhn_to_mnist(svhn).view(-1, *mnist_shape), (2, 2, 2, 2),
                                        mode='constant', value=0)
                mnist_to_svhn = vae.mnist_to_svhn(mnist)
                mnist = F.pad(mnist.view(-1, *mnist_shape), (2, 2, 2, 2),
                                mode='constant', value=0)
                svhn_mnist = torch.cat([svhn.expand(-1, 3, -1, -1),
                                        svhn_to_mnist.expand(-1, 3, -1, -1)], dim=-1)
                mnist_svhn = torch.cat([mnist.expand(-1, 3, -1, -1),
                                        mnist_to_svhn.expand(-1, 3, -1, -1)], dim=-1)
                svhn_grid = make_grid(svhn[:20], nrow=1)
                recon_svhn = torch.cat([svhn, recon_svhn], dim=-1)
                recon_mnist = torch.cat([mnist, recon_mnist], dim=-1)

                save_image(make_grid(svhn_mnist, nrow=8), os.path.join(args.data_dir, 'img/svhn_mnist_%i.png' % epoch))
                save_image(make_grid(mnist_svhn, nrow=8), os.path.join(args.data_dir, 'img/mnist_svhn_%i.png' % epoch))
                save_image(make_grid(recon_svhn, nrow=8), os.path.join(args.data_dir, 'img/recon_data.png'))
                save_image(make_grid(recon_mnist, nrow=8), os.path.join(args.data_dir, 'img/recon_mnist.png'))

                figs = vae.tsne_plot(data_loaders['test'], device, args.data_dir, epoch, 'inf')

        
        print("[Epoch %03d] Sup Loss %.3f, Unsup Loss %.3f" % 
                (epoch, epoch_losses_sup, epoch_losses_unsup))
        vae.save_models(args.data_dir)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optim.state_dict(),
            }, os.path.join(args.data_dir, 'optim.pt'))

    vae.save_models(args.data_dir)

def parser_args(parser):
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-n', '--num-epochs', default=500, type=int,
                        help="number of epochs")
    parser.add_argument('-sup', '--sup-frac', default=1.0,
                        type=float, help="supervised fractional amount of the data i.e. "
                                         "how many of the images have supervised labels."
                                         "Should be a multiple of train_size / batch_size")
    parser.add_argument('--missing', default=None, help='a|b a is SVHN missing, b is MNIST')
    parser.add_argument('-zd', '--z_dim', default=64, type=int,
                        help="latent size")
    parser.add_argument('-lr', '--learning-rate', default=5e-4, type=float)
    parser.add_argument('-bs', '--batch-size', default=128, type=int)
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data path')
    parser.add_argument('--seed', type=int, default=1)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser)
    args = parser.parse_args()
    
    set_seed(args.seed)

    if args.sup_frac < 1.0:
        assert args.missing is not None, "Set missing modality for semi-sup"

    run_name = ('_').join(["Sup", str(args.sup_frac),
                            "Missing", str(args.missing)])
    args.data_dir = os.path.join(args.data_dir, 'runs', run_name)

    if os.path.isdir(args.data_dir):
        shutil.rmtree(args.data_dir)

    os.makedirs(os.path.join(args.data_dir, "img"))
    main(args)

