import argparse
import datetime
import gc
import imageio
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import tqdm

from cv2 import putText

from moving_mnist import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='.',
                    help="Path where 'train-images-idx3-ubyte.gz' can be found") # http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
parser.add_argument('--save_path', type=str, default='./ODE_MMNIST_EXP1')
parser.add_argument('--n_vids', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
# Model, training
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_res_blocks', type=int, default=4)
parser.add_argument('--res_ch', nargs='+', type=int, default=[8, 16, 32, 32])
# parser.add_argument('--obs_dim', type=int, default=256)
parser.add_argument('--nhidden', type=int, default=128)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--skip_level', type=int, choices=[0, 1, 2, 3, 4], default=0)
parser.add_argument('--gpu', type=int, default=0)
# Data
parser.add_argument('--num_digits', type=int, default=1)
parser.add_argument('--n_frames_input', type=int, default=10)
parser.add_argument('--n_frames_output', type=int, default=10)
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--im_ch', type=int, default=1)

parser.add_argument('--adjoint', type=eval, default=False)

parser.add_argument('--model_save_step', type=int, default=100)
parser.add_argument('--vis_step', type=int, default=100)
parser.add_argument('--vis_n_vids', type=int, default=50, help="How many videos to visualize, must be <= batch_size")
parser.add_argument('--n_epochs', type=int, default=1002)
parser.add_argument('--num_workers', type=int, default=8)

args = parser.parse_args()

assert args.vis_n_vids <= args.batch_size, "ERROR: vis_n_vids must be <= batch_size! Given vis_n_vids=" + str(args.vis_n_vids) + " ; batch_size=" + str(args.batch_size)
assert len(args.res_ch) == args.n_res_blocks, "ERROR: # of elements in res_ch must be equal to n_res_blocks! Given n_res_blocks = " + args.n_res_blocks + " ; res_ch = " + str(args.res_ch) + " ; len(res_ch) = " + str(len(args.res_ch))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def ode_collate_fn(batch):
    frames_input, frames_output = torch.stack([b[0] for b in batch]).float().mul(2.).sub(1.), torch.stack([b[1] for b in batch]).float().mul(2.).sub(1.)
    frames_all = torch.cat((frames_input, frames_output), dim=1)
    times_all = torch.from_numpy(np.linspace(0., 1., num=frames_all.shape[1])).float()
    times_input = times_all[:frames_input.shape[1]]
    return [frames_all, frames_input, times_all, times_input]


def moving_mnist_ode_data_loader(dset_path, num_objects=[1], batch_size=100,
                                 n_frames_input=5, n_frames_output=5,
                                 num_workers=8):
    transform = transforms.Compose([ToTensor()])
    dset = MovingMNIST(dset_path, True, n_frames_input, n_frames_output, num_objects, transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=ode_collate_fn,
                              num_workers=num_workers, pin_memory=True)
    return dloader


class LatentODEfunc(nn.Module):
    
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=2, mode='downsample'):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if mode == 'downsample':
            self.resample = conv1x1(inplanes, planes, stride)
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.resample = nn.ConvTranspose2d(inplanes, planes, stride, stride)
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, stride, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        shortcut = self.resample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.norm2(out)))
        return out + shortcut


class EncoderRNN(nn.Module):

    def __init__(self, n_res_blocks=4, res_ch=[16, 32, 64, 64],
                 nhidden=256, latent_dim=128, skip_level=0,
                 imsize=64, im_ch=1):
        super(EncoderRNN, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.res_ch = res_ch
        # self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.latent_dim = latent_dim
        self.skip_level = skip_level
        self.imsize = imsize
        self.im_ch = im_ch

        self.res_blocks = nn.ModuleList()
        in_ch = self.im_ch
        for i in range(self.n_res_blocks):
            out_ch = self.res_ch[i]
            self.res_blocks.append(ResBlock(in_ch, out_ch, 2, 'downsample'))
            in_ch = out_ch

        # self.res_block1 = ResBlock(1, 16, 2, 'downsample')      # 32x32
        # self.res_block2 = ResBlock(16, 32, 2, 'downsample')     # 16x16
        # self.res_block3 = ResBlock(32, 64, 2, 'downsample')     # 8x8
        # self.res_block4 = ResBlock(64, 64, 2, 'downsample')     # 4x4; 64*4*4 = 1024 = obs_dim

        self.obs_dim = (self.imsize//2**self.n_res_blocks)**2 * self.res_ch[-1]

        self.i2h = nn.Linear(self.obs_dim + self.nhidden, self.nhidden)
        self.h2o = nn.Linear(self.nhidden, self.latent_dim * 2)

    def forward(self, x, h):
        bs = x.shape[0]
        feats = None
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            if i+1 == self.skip_level:
                feats = x

        x = x.view(bs, -1)
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h, feats

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=128, nhidden=256, # obs_dim=1024,
                 n_res_blocks=4, res_ch=[64, 64, 32, 16], skip_level=0,
                 imsize=64, im_ch=1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        # self.obs_dim = obs_dim
        self.n_res_blocks = n_res_blocks
        self.res_ch = res_ch
        self.skip_level = skip_level
        self.imsize = imsize
        self.im_ch = im_ch

        self.init_imsize = self.imsize//2**(self.n_res_blocks)
        self.obs_dim = int((self.init_imsize)**2 * self.res_ch[0])

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.latent_dim, self.nhidden)
        self.fc2 = nn.Linear(self.nhidden, self.obs_dim)

        self.res_blocks = nn.ModuleList()
        in_ch = self.res_ch[0]
        out_ch = self.res_ch[1] if len(self.res_ch) > 1 else self.im_ch
        for i in range(self.n_res_blocks):
            in_ch = res_ch[i]
            out_ch = self.res_ch[i+1] if i < self.n_res_blocks-1 else self.im_ch
            self.res_blocks.append(ResBlock(in_ch*(1 + int(self.skip_level==(self.n_res_blocks-i))), out_ch, 2, 'upsample'))

        # self.res_block1 = ResBlock(64*(1 + int(self.skip_level==4)), 64, 2, 'upsample')
        # self.res_block2 = ResBlock(64*(1 + int(self.skip_level==3)), 32, 2, 'upsample')
        # self.res_block3 = ResBlock(32*(1 + int(self.skip_level==2)), 16, 2, 'upsample')
        # self.res_block4 = ResBlock(16*(1 + int(self.skip_level==1)), 1, 2, 'upsample')
        self.tanh = nn.Tanh()

    def forward(self, z, feats=None):
        bs = z.shape[0]
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(-1, self.res_ch[0], self.init_imsize, self.init_imsize)
        
        for i, res_block in enumerate(self.res_blocks):
            if self.skip_level == self.n_res_blocks - i:
                out = torch.cat((out, feats), dim=1)
            out = res_block(out)

        out = self.tanh(out)
        out = out.view(bs, -1, self.im_ch, self.imsize, self.imsize)
        return out


def load_model(ckpt_path, func, enc, dec):
    checkpoint = torch.load(ckpt_path)
    func.load_state_dict(checkpoint['func_state_dict'])
    enc.load_state_dict(checkpoint['enc_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    return func, enc, dec


start = 0.
stop = 1.
noise_std = .3
device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Data
print("Loading data")
dl = moving_mnist_ode_data_loader(args.data_path, num_objects=[args.num_digits], batch_size=args.n_vids,
                                    n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output,
                                    num_workers=args.num_workers)
orig_trajs, samp_trajs, orig_ts, samp_ts = next(iter(dl))
samp_trajs, samp_ts = samp_trajs.to(device), samp_ts.to(device)
orig_trajs_vis = orig_trajs[:args.vis_n_vids]
orig_trajs_vis_gpu = orig_trajs_vis.clone().to(device)

func = LatentODEfunc(args.latent_dim, args.nhidden).to(device)
enc = EncoderRNN(args.n_res_blocks, args.res_ch, args.nhidden, args.latent_dim, args.skip_level, args.imsize, args.im_ch).to(device)
dec = Decoder(args.latent_dim, args.nhidden, args.n_res_blocks, args.res_ch[::-1], args.skip_level, args.imsize, args.im_ch).to(device)
params = (list(func.parameters()) + list(dec.parameters()) + list(enc.parameters()))
optimizer = optim.Adam(params, lr=args.lr)
