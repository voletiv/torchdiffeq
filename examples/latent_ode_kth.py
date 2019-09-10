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

from kth_dataset import KTH_Dataset

# python latent_ode_kth.py --h5_path /home/voletivi/scratch/Datasets/KTH/kth.h5 --save_path /home/voletivi/scratch/ode/ODE_KTH --n_res_blocks 4 --res_ch 4 8 16 16 --nhidden 128 --latent_dim 16  --batch_size 128 --vis_step 100 --vis_n_vids 50 --num_workers 8 --seed 0


parser = argparse.ArgumentParser()
parser.add_argument('--h5_path', type=str, help="Path of 'kth.h5'")
parser.add_argument('--save_path', type=str, default='./ODE_KTH_EXP1')
parser.add_argument('--classes', type=str, nargs='+', default=None, choices=[None, 'walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping'])
parser.add_argument('--seed', type=int, default=0)
# Model, training
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_res_blocks', type=int, default=4)
parser.add_argument('--res_ch', nargs='+', type=int, default=[8, 16, 32, 32])
# parser.add_argument('--obs_dim', type=int, default=256)
parser.add_argument('--nhidden', type=int, default=128)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--skip_level', type=int, choices=[0, 1, 2, 3, 4], default=0)
parser.add_argument('--gpu', type=int, default=0)
# Data
parser.add_argument('--n_frames_cond', type=int, default=10)
parser.add_argument('--n_frames_pred', type=int, default=10)
parser.add_argument('--n_frames_future', type=int, default=30)
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--im_ch', type=int, default=1)

parser.add_argument('--adjoint', type=eval, default=False)

# parser.add_argument('--time_start', type=float, default=0)
parser.add_argument('--time_stop_cond', type=float, default=0.5)
parser.add_argument('--noise_std', type=float, default=0.3)

parser.add_argument('--vis_step', type=int, default=50)
parser.add_argument('--vis_n_vids', type=int, default=50, help="How many videos to visualize, must be <= batch_size")
parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()

args.time_start = 0.
assert args.vis_n_vids <= args.batch_size, "ERROR: vis_n_vids must be <= batch_size! Given vis_n_vids=" + str(args.vis_n_vids) + " ; batch_size=" + str(args.batch_size)
assert len(args.res_ch) == args.n_res_blocks, "ERROR: # of elements in res_ch must be equal to n_res_blocks! Given n_res_blocks = " + args.n_res_blocks + " ; res_ch = " + str(args.res_ch) + " ; len(res_ch) = " + str(len(args.res_ch))

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


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


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mem_check():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("Mem:", process.memory_info().rss/1024/1024/1024, "GB")


if __name__ == '__main__':

    np.random.seed (args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.save_path):
        args.save_path = os.path.join(os.path.dirname(args.save_path),
            f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.path.basename(args.save_path)}_')
        if args.classes:
            for l in args.classes:
                args.save_path += l + '_'
        args.save_path += 'res_'
        for i in range(args.n_res_blocks):
            args.save_path += f'{args.res_ch[i]}_'
        args.save_path += f'hid{args.nhidden}_z{args.latent_dim}_skip{args.skip_level}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}'
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'samples'))

    print(args)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # DATA
    print("Loading data")
    # Time steps
    times_pred = torch.tensor(np.linspace(args.time_start, args.time_stop_cond, args.n_frames_pred))
    times_predf = torch.cat([times_pred, torch.tensor(np.linspace(args.time_stop_cond + (args.time_stop_cond - args.time_start)/args.n_frames_pred, args.time_stop_cond + (args.time_stop_cond - args.time_start)/args.n_frames_pred*args.n_frames_future, args.n_frames_future))])
    times_pred, times_predf = times_pred.to(device), times_predf.to(device)
    # Train
    train_ds = KTH_Dataset(args.h5_path, val=False, classes=args.classes,
                            n_frames_cond=args.n_frames_cond, n_frames_pred=args.n_frames_pred, n_frames_future=args.n_frames_future,
                            time_start=0, time_stop_cond=args.time_stop_cond)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: train_ds.kth_collate_fn(batch))
    fixed_train_vids_cond, fixed_train_vids_pred, fixed_train_vids_future, _ = next(iter(train_dl))
    fixed_train_vids_cond, fixed_train_vids_predf = fixed_train_vids_cond[:args.vis_n_vids].to(device), torch.cat([fixed_train_vids_pred[:args.vis_n_vids], fixed_train_vids_future[:args.vis_n_vids]], dim=1)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda batch: train_ds.kth_collate_fn(batch))
    # Val
    val_ds = KTH_Dataset(args.h5_path, val=True, classes=args.classes,
                            n_frames_cond=args.n_frames_cond, n_frames_pred=args.n_frames_pred, n_frames_future=args.n_frames_future,
                            time_start=0, time_stop_cond=args.time_stop_cond)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=(len(val_ds) > args.batch_size), collate_fn=lambda batch: val_ds.kth_collate_fn(batch))
    fixed_val_vids_cond, fixed_val_vids_pred, fixed_val_vids_future, _ = next(iter(val_dl))
    fixed_val_vids_cond, fixed_val_vids_predf = fixed_val_vids_cond[:args.vis_n_vids].to(device), torch.cat([fixed_val_vids_pred[:args.vis_n_vids], fixed_val_vids_future[:args.vis_n_vids]], dim=1)
    fixed_val_vids_predf_gpu = fixed_val_vids_predf.clone().to(device)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=lambda batch: val_ds.kth_collate_fn(batch))

    # Model
    print("Making models")
    func = LatentODEfunc(args.latent_dim, args.nhidden).to(device)
    enc = EncoderRNN(args.n_res_blocks, args.res_ch, args.nhidden, args.latent_dim, args.skip_level, args.imsize, args.im_ch).to(device)
    dec = Decoder(args.latent_dim, args.nhidden, args.n_res_blocks, args.res_ch[::-1], args.skip_level, args.imsize, args.im_ch).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(enc.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)

    # Vars
    loss_meter = RunningAverageMeter(momentum=0.99)
    losses = []
    losses_ma = []
    val_loss_meter_pred = RunningAverageMeter(momentum=0.9)
    val_losses_pred = []
    val_losses_ma_pred = []
    val_loss_meter_future = RunningAverageMeter(momentum=0.9)
    val_losses_future = []
    val_losses_ma_future = []

    if args.save_path is not None:
        ckpt_path = os.path.join(args.save_path, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            print("Loading model ckpt from", args.save_path)
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            enc.load_state_dict(checkpoint['enc_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    log_file_name = os.path.join(args.save_path, 'log.txt')
    log_file = open(log_file_name, "wt")
    log_file.write(str(args) + '\n')
    log_file.flush()

    noise_std_ = torch.zeros(args.batch_size, args.n_frames_pred, args.im_ch, args.imsize, args.imsize) + args.noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    noise_std_val_ = torch.zeros(args.vis_n_vids, args.n_frames_pred + args.n_frames_future, args.im_ch, args.imsize, args.imsize) + args.noise_std
    noise_logvar_val = 2. * torch.log(noise_std_val_).to(device)

    try:
        print("Starting training...")
        print("n_batches", len(train_dl))

        for epoch in range(1, args.n_epochs + 1):
            total_loss = 0

            # TRAIN
            for batch_vids_cond, batch_vids_pred, _, _ in train_dl:
                # print("epoch", epoch)
                optimizer.zero_grad()

                batch_vids_cond, batch_vids_pred = batch_vids_cond.to(device), batch_vids_pred.to(device)

                h = enc.initHidden(args.batch_size).to(device)
                for t in range(args.n_frames_cond):
                    obs = batch_vids_cond[:, t]
                    out, h, feats = enc.forward(obs, h)

                qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean    # Bxdim

                # forward in time and solve ode for predictions
                # print("doing ode")
                pred_z = odeint(func, z0, times_pred).permute(1, 0, 2)     # BxTxdim
                # print("decoding after ode")
                pred_x = dec(pred_z, feats)    # BxTx1x64x64

                # compute loss
                # print("computing loss")
                logpx = log_normal_pdf(batch_vids_pred, pred_x, noise_logvar).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                loss = torch.mean(-logpx + analytic_kl, dim=0)
                # print("doing loss.backward()")
                loss.backward()
                # print("doing optimizer.step()")
                optimizer.step()
                total_loss += loss.item()

                del loss, out, h, pred_z, pred_x

            loss_meter.update(total_loss/len(train_dl))
            losses.append(total_loss/len(train_dl))
            losses_ma.append(loss_meter.avg)

            log = 'Epoch: {}, running avg loss: {:.4f}\n'.format(epoch, loss_meter.avg)
            print(args.save_path)
            print(log)
            log_file.write(log)
            log_file.flush()

            # VISUALIZE
            if epoch % args.vis_step == 0:

                # Sampling from TRAIN data
                # print("Sampling")
                with torch.no_grad():
                    h = enc.initHidden(args.vis_n_vids).to(device)
                    for t in range(args.n_frames_cond):
                        obs = fixed_train_vids_cond[:, t]
                        out, h, feats = enc.forward(obs, h)
                    qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
                    epsilon = torch.randn(qz0_mean.size()).to(device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                    pred_z = odeint(func, z0, times_predf).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z = dec(pred_z, feats)    # BxTx1x64x64

                xs_dec_pred_z = xs_dec_pred_z.cpu()

                frames = []
                for t in range(xs_dec_pred_z.shape[1]):
                    xs_t = fixed_train_vids_predf[:, t]         # Bx1x64x64
                    xs_dec_pred_z_t = xs_dec_pred_z[:, t]       # Bx1x64x64
                    frame = torch.cat([xs_t, torch.ones(args.vis_n_vids, 1, 64, 2),
                                       xs_dec_pred_z_t], dim=-1)
                    gif_frame = vutils.make_grid(frame, nrow=10, padding=8, pad_value=1).permute(1, 2, 0).add(1.).mul(0.5).numpy()
                    frames.append((putText(np.concatenate((np.ones((40, gif_frame.shape[1], gif_frame.shape[2])), gif_frame), axis=0), f"time = {t+1}", (8, 30), 0, 1, (0,0,0), 4)*255).astype('uint8'))
                    del frame, gif_frame

                imageio.mimwrite(os.path.join(args.save_path, 'samples', f'train_vis_{epoch:06d}.gif'), frames, fps=4)
                del xs_dec_pred_z, frames, h, out

                # Sampling from VAL data
                # print("Sampling")
                with torch.no_grad():
                    h = enc.initHidden(args.vis_n_vids).to(device)
                    for t in range(args.n_frames_cond):
                        obs = fixed_val_vids_predf_gpu[:, t]
                        out, h, feats = enc.forward(obs, h)
                    qz0_mean_val, qz0_logvar_val = out[:, :args.latent_dim], out[:, args.latent_dim:]
                    epsilon = torch.randn(qz0_mean.size()).to(device)
                    z0_val = epsilon * torch.exp(.5 * qz0_logvar_val) + qz0_mean_val

                    pred_z_val = odeint(func, z0_val, times_predf).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z_val = dec(pred_z_val, feats)  # BxTx1x64x64

                logpx_val_pred = log_normal_pdf(fixed_val_vids_predf_gpu[:, :args.n_frames_pred], xs_dec_pred_z_val[:, :args.n_frames_pred], noise_logvar_val[:, :args.n_frames_pred]).sum(-1).sum(-1).sum(-1).sum(-1).mean()

                pz0_mean_val = pz0_logvar_val = torch.zeros(z0_val.size()).to(device)
                analytic_kl_val = normal_kl(qz0_mean_val, qz0_logvar_val,
                                            pz0_mean_val, pz0_logvar_val).sum(-1)
                val_loss_pred = torch.mean(-logpx_val_pred + analytic_kl_val, dim=0).item()

                val_loss_meter_pred.update(val_loss_pred)
                val_losses_pred.append(val_loss_pred)
                val_losses_ma_pred.append(val_loss_meter_pred.avg)

                logpx_val_future = log_normal_pdf(fixed_val_vids_predf_gpu[:, args.n_frames_pred:], xs_dec_pred_z_val[:, args.n_frames_pred:], noise_logvar_val[:, args.n_frames_pred:]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                val_loss_future = -logpx_val_future.item()
                val_loss_meter_future.update(val_loss_future)
                val_losses_future.append(val_loss_future)
                val_losses_ma_future.append(val_loss_meter_future.avg)

                xs_dec_pred_z_val = xs_dec_pred_z_val.cpu()

                frames_val = []
                for t in range(xs_dec_pred_z_val.shape[1]):
                    xs_t = fixed_val_vids_predf[:, t]              # Bx64x64
                    xs_dec_pred_z_t = xs_dec_pred_z_val[:, t]      # Bx64x64
                    frame = torch.cat([xs_t, torch.ones(args.vis_n_vids, args.im_ch, args.imsize, 2),
                                       xs_dec_pred_z_t], dim=-1)
                    gif_frame = vutils.make_grid(frame, nrow=10, padding=8, pad_value=1).permute(1, 2, 0).add(1.).mul(0.5).numpy()
                    frames_val.append((putText(np.concatenate((np.ones((40, gif_frame.shape[1], gif_frame.shape[2])), gif_frame), axis=0), f"time = {t+1}", (8, 30), 0, 1, (0,0,0), 4)*255).astype('uint8'))
                    del frame, gif_frame

                imageio.mimwrite(os.path.join(args.save_path, 'samples', f'val_vis_{epoch:06d}.gif'), frames_val, fps=4)
                del xs_dec_pred_z_val, val_loss_pred, val_loss_future, frames_val

                # Plot
                plt.plot(np.arange(1, epoch+1), losses, '--', c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(1, epoch+1), losses_ma, c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_pred, '--', c='C1', alpha=0.5, label="val_loss_pred")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_pred, c='C1', alpha=0.5, label="val_loss_ma_pred")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_future, '--', c='C2', alpha=0.5, label="val_loss_future")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_future, c='C2', alpha=0.5, label="val_loss_ma_future")
                plt.legend()
                plt.yscale("symlog")
                plt.xlabel("Epochs")
                # plt.title("Losses")
                plt.savefig(os.path.join(args.save_path, "loss.png"), bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

                # Garbage collection
                gc.collect()
                mem_check()

    except KeyboardInterrupt:
        print("Ctrl+C!\n")

    ckpt_path = os.path.join(args.save_path, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'enc_state_dict': enc.state_dict(),
        'dec_state_dict': dec.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    log = 'Stored ckpt at {}\n'.format(ckpt_path)
    print(log)
    log_file.write(log)
    log_file.flush()

    log = 'Training complete after {} epochs.\n'.format(epoch)
    print(log)
    log_file.write(log)
    log_file.flush()
