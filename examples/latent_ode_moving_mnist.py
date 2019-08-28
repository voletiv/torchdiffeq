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

# python latent_ode_moving_mnist.py --num_of_vids 1000 --batch_size 100 --save_path /home/voletivi/scratch/ode/ODE_SKIPcat_l1 --skip_level 1 --vis_step 50 --vis_n_vids 50

# for i in tqdm.tqdm(range(10000)):
#     this_dir = '/home/voletiv/Datasets/MyMovingMNIST/{:05d}'.format(i)
#     os.makedirs(this_dir)
#     for j in range(20):
#         im = orig_trajs[i, j, 0].numpy()
#         im = (im + 1.)/2.*255.
#         im = im.astype('uint8')
#         imageio.imwrite(os.path.join(this_dir, '{:02d}.png'.format(j+1)), im)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='.',
                    help="Path where 'train-images-idx3-ubyte.gz' can be found") # http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
parser.add_argument('--save_path', type=str, default='./ODE_MMNIST_EXP1')
parser.add_argument('--num_of_vids', type=int, default=1000)
parser.add_argument('--skip_level', type=int, choices=[1, 2, 3, 4], default=4)
# Training
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--obs_dim', type=int, default=1024)
parser.add_argument('--nhidden', type=int, default=256)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
# Data
parser.add_argument('--num_digits', type=int, default=1)
parser.add_argument('--n_frames_input', type=int, default=10)
parser.add_argument('--n_frames_output', type=int, default=10)
parser.add_argument('--imsize', type=int, default=64)
parser.add_argument('--im_ch', type=int, default=1)

parser.add_argument('--adjoint', type=eval, default=False)

parser.add_argument('--vis_step', type=int, default=50)
parser.add_argument('--vis_n_vids', type=int, default=50, help="How many videos to visualize, must be <= batch_size")
parser.add_argument('--niters', type=int, default=10000)

args = parser.parse_args()

assert args.vis_n_vids <= args.batch_size, "ERROR: vis_n_vids must be <= batch_size! Given vis_n_vids=" + str(args.vis_n_vids) + " ; batch_size=" + str(args.batch_size)

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
                                 n_workers=8):
    transform = transforms.Compose([ToTensor()])
    dset = MovingMNIST(dset_path, True, n_frames_input, n_frames_output, num_objects, transform)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=ode_collate_fn,
                              num_workers=n_workers, pin_memory=True)
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


# class EncoderRNN(nn.Module):

#     def __init__(self, latent_dim=4, obs_dim=16, nhidden=25):
#         super(EncoderRNN, self).__init__()
#         self.latent_dim = latent_dim
#         self.obs_dim = obs_dim
#         self.nhidden = nhidden
#         self.res_block1 = ResBlock(1, 16, 2, 'downsample')    # 32x32
#         self.res_block2 = ResBlock(16, 32, 2, 'downsample')    # 16x16
#         self.res_block3 = ResBlock(32, 64, 2, 'downsample')    # 8x8
#         self.res_block4 = ResBlock(64, 2, 2, 'downsample')    # 4x4; 2*4*4 = 32 = 2*obs_dim
#         self.i2h = nn.Linear(2*obs_dim + nhidden, nhidden)
#         self.h2o = nn.Linear(nhidden, latent_dim * 2)

#     def forward(self, x, h):
#         bs = x.shape[0]
#         x = self.res_block4(self.res_block3(self.res_block2(self.res_block1(x))))
#         x = x.view(bs, -1)
#         combined = torch.cat((x, h), dim=1)
#         h = torch.tanh(self.i2h(combined))
#         out = self.h2o(h)
#         return out, h

#     def initHidden(self, batch_size):
#         return torch.zeros(batch_size, self.nhidden)


class Encoder(nn.Module):

    def __init__(self, latent_dim=128, obs_dim=1024, nhidden=256, skip_level=1):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.skip_level = skip_level
        self.res_block1 = ResBlock(1, 16, 2, 'downsample')      # 32x32 <-- 64x64
        self.res_block2 = ResBlock(16, 32, 2, 'downsample')     # 16x16
        self.res_block3 = ResBlock(32, 64, 2, 'downsample')     # 8x8
        self.res_block4 = ResBlock(64, 64, 2, 'downsample')     # 4x4; 4*4*64 = 1024 = obs_dim
        self.i2h = nn.Linear(obs_dim, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        # bs = x.shape[0]
        x1 = self.res_block1(x)     # 32x32
        x2 = self.res_block2(x1)    # 16x16
        x3 = self.res_block3(x2)    # 8x8
        x4 = self.res_block4(x3)    # 4x4
        x = x4.view(-1, self.obs_dim)
        h = torch.tanh(self.i2h(x))
        out = self.h2o(h)
        if self.skip_level == 1:
            feats = x1
        elif self.skip_level == 2:
            feats = x2
        elif self.skip_level == 3:
            feats = x3
        elif self.skip_level == 4:
            feats = x4
        return out, feats


class Decoder(nn.Module):

    def __init__(self, latent_dim=128, obs_dim=1024, nhidden=256, skip_level=1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.skip_level = skip_level
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)
        self.res_block1 = ResBlock(64*(1 + int(self.skip_level==4)), 64, 2, 'upsample')
        self.res_block2 = ResBlock(64*(1 + int(self.skip_level==3)), 32, 2, 'upsample')
        self.res_block3 = ResBlock(32*(1 + int(self.skip_level==2)), 16, 2, 'upsample')
        self.res_block4 = ResBlock(16*(1 + int(self.skip_level==1)), 1, 2, 'upsample')
        self.tanh = nn.Tanh()

    def forward(self, z, feats):
        bs = z.shape[0]
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(-1, 64, 4, 4)
        if self.skip_level == 4:
            out = torch.cat((out, feats), dim=1)
        out = self.res_block1(out)
        if self.skip_level == 3:
            out = torch.cat((out, feats), dim=1)
        out = self.res_block2(out)
        if self.skip_level == 2:
            out = torch.cat((out, feats), dim=1)
        out = self.res_block3(out)
        if self.skip_level == 1:
            out = torch.cat((out, feats), dim=1)
        out = self.res_block4(out)
        out = self.tanh(out)
        out = out.view(bs, -1, 1, 64, 64)
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
        args.save_path = os.path.join(os.path.dirname(args.save_path), '{0:%Y%m%d_%H%M%S}_{1}'.format(datetime.datetime.now(), os.path.basename(args.save_path)))
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'samples'))

    start = 0.
    stop = 1.
    noise_std = .3
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Data
    print("Loading data")
    dl = moving_mnist_ode_data_loader(args.data_path, num_objects=[args.num_digits], batch_size=args.num_of_vids,
                                        n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output,
                                        n_workers=8)
    orig_trajs, samp_trajs, orig_ts, samp_ts = next(iter(dl))
    samp_trajs, samp_ts = samp_trajs.to(device), samp_ts.to(device)
    orig_trajs_vis = orig_trajs[:args.vis_n_vids]
    orig_trajs_vis_gpu = orig_trajs_vis.clone().to(device)

    # Val data
    print("Loading val data")
    orig_trajs_val, _, orig_ts_val, _ = next(iter(dl))
    orig_trajs_val = orig_trajs_val[:args.vis_n_vids]
    orig_trajs_val_gpu = orig_trajs_val.clone().to(device)

    # Model
    print("Making models")
    func = LatentODEfunc(args.latent_dim, args.nhidden).to(device)
    # enc = EncoderRNN(latent_dim, obs_dim, rnn_nhidden).to(device)
    enc = Encoder(args.latent_dim, args.obs_dim, args.nhidden, args.skip_level).to(device)
    dec = Decoder(args.latent_dim, args.obs_dim, args.nhidden, args.skip_level).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(enc.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)

    # Vars
    loss_meter = RunningAverageMeter(momentum=0.99)
    losses = []
    losses_ma = []
    val_loss_meter_input = RunningAverageMeter(momentum=0.9)
    val_losses_input = []
    val_losses_ma_input = []
    val_loss_meter_output = RunningAverageMeter(momentum=0.9)
    val_losses_output = []
    val_losses_ma_output = []
    ts_pos = np.linspace(start, stop, num=args.n_frames_input+args.n_frames_output)
    ts_pos = torch.from_numpy(ts_pos).float().to(device)

    if args.save_path is not None:
        ckpt_path = os.path.join(args.save_path, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            print("Loading model ckpt from", args.save_path)
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            enc.load_state_dict(checkpoint['enc_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    log_file_name = os.path.join(args.save_path, 'log.txt')
    log_file = open(log_file_name, "wt")

    noise_std_ = torch.zeros(args.batch_size, args.n_frames_input, args.im_ch, args.imsize, args.imsize) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    noise_std_z = torch.zeros(args.batch_size, args.n_frames_input, args.latent_dim) + noise_std
    noise_logvar_z = 2. * torch.log(noise_std_z).to(device)

    try:
        print("Starting training...")
        n_batches = args.num_of_vids//args.batch_size
        print("n_batches", n_batches)

        vid_ids = np.arange(args.num_of_vids)

        for itr in range(1, args.niters + 1):
            total_loss = 0
            np.random.shuffle(vid_ids)

            # TRAIN
            for b in range(n_batches):
                # print("itr", itr)
                optimizer.zero_grad()
                z, feats = enc(samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size]].view(-1, args.im_ch, args.imsize, args.imsize))     # B*Txdim
                z = z.view(args.batch_size, -1, args.latent_dim)    # BxTxdim
                feats = feats.view(args.batch_size, -1, *feats.shape[1:])[:, 0:1].expand(args.batch_size, feats.shape[0]//args.batch_size, *feats.shape[1:]).contiguous().view(-1, *feats.shape[1:])
                # forward in time and solve ode for reconstructions
                # print("doing ode")
                pred_z = odeint(func, z.permute(1, 0, 2)[0], samp_ts).permute(1, 0, 2)     # BxTxdim
                # print("decoding after ode")
                pred_x = dec(pred_z, feats)    # BxTx1x64x64

                # compute loss
                # print("computing loss")
                logpx = log_normal_pdf(samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size]], pred_x, noise_logvar).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                logpx += log_normal_pdf(z, pred_z, noise_logvar_z).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                # print("doing loss.backward()")
                loss = -logpx
                loss.backward()
                # print("doing optimizer.step()")
                optimizer.step()
                total_loss += loss.item()

            loss_meter.update(total_loss/n_batches)
            losses.append(total_loss/n_batches)
            losses_ma.append(loss_meter.avg)

            log = 'Iter: {}, running avg loss: {:.4f}\n'.format(itr, loss_meter.avg)
            print(log)
            log_file.write(log)
            log_file.flush()

            # VISUALIZE
            if itr % args.vis_step == 0:

                # Sampling from TRAIN data
                # print("Sampling")
                with torch.no_grad():
                    z, feats = enc(orig_trajs_vis_gpu.view(-1, args.im_ch, args.imsize, args.imsize))   # B*Txdim
                    z = z.view(args.vis_n_vids, -1, args.latent_dim)    # BxTxdim
                    feats = feats.view(args.vis_n_vids, -1, *feats.shape[1:])[:, 0:1].expand(args.vis_n_vids, feats.shape[0]//args.vis_n_vids, *feats.shape[1:]).contiguous().view(-1, *feats.shape[1:])
                    xs_dec_z = dec(z, feats)    # BxTx1x64x64
                    pred_z = odeint(func, z.permute(1, 0, 2)[0], ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z = dec(pred_z, feats)    # BxTx1x64x64

                xs_dec_z = xs_dec_z.cpu()
                xs_dec_pred_z = xs_dec_pred_z.cpu()

                frames = []
                for t in range(xs_dec_pred_z.shape[1]):
                    xs_t = orig_trajs_vis[:, t]                        # Bx1x64x64
                    xs_dec_z_t = xs_dec_z[:, t]                 # Bx1x64x64
                    xs_dec_pred_z_t = xs_dec_pred_z[:, t]       # Bx1x64x64
                    frame = torch.cat([xs_t, torch.ones(args.vis_n_vids, 1, 64, 2),
                                       xs_dec_z_t, torch.ones(args.vis_n_vids, 1, 64, 2),
                                       xs_dec_pred_z_t], dim=-1)
                    gif_frame = vutils.make_grid(frame, nrow=5, padding=8, pad_value=1).permute(1, 2, 0).add(1.).mul(0.5).numpy()
                    frames.append((putText(np.concatenate((np.ones((40, gif_frame.shape[1], gif_frame.shape[2])), gif_frame), axis=0), f"time = {t+1}", (8, 30), 0, 1, (0,0,0), 4)*255).astype('uint8'))
                    del frame, gif_frame

                imageio.mimwrite(os.path.join(args.save_path, 'samples', f'train_vis_{itr:06d}.gif'), frames, fps=4)
                del xs_dec_z, xs_dec_pred_z, frames

                # Sampling from VAL data
                # print("Sampling")
                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    z_val, feats_val = enc(orig_trajs_val_gpu.view(-1, args.im_ch, args.imsize, args.imsize))   # BxTxdim
                    z_val = z_val.view(args.vis_n_vids, -1, args.latent_dim)    # BxTxdim
                    feats_val = feats_val.view(args.vis_n_vids, -1, *feats_val.shape[1:])[:, 0:1].expand(args.vis_n_vids, feats_val.shape[0]//args.vis_n_vids, *feats_val.shape[1:]).contiguous().view(-1, *feats_val.shape[1:])
                    xs_dec_z_val = dec(z_val, feats_val)    # BxTx1x64x64
                    pred_z_val = odeint(func, z_val.permute(1, 0, 2)[0], ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z_val = dec(pred_z_val, feats_val)  # BxTx1x64x64

                logpx_val_input = log_normal_pdf(orig_trajs_val_gpu[:, :args.n_frames_input], xs_dec_pred_z_val[:, :args.n_frames_input], noise_logvar[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                logpx_val_input += log_normal_pdf(z_val[:, :args.n_frames_input], pred_z_val[:, :args.n_frames_input], noise_logvar_z[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                val_loss_input = -logpx_val_input.item()
                val_loss_meter_input.update(val_loss_input)
                val_losses_input.append(val_loss_input)
                val_losses_ma_input.append(val_loss_meter_input.avg)

                logpx_val_output = log_normal_pdf(orig_trajs_val_gpu[:, args.n_frames_input:], xs_dec_pred_z_val[:, args.n_frames_input:], noise_logvar[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                logpx_val_output += log_normal_pdf(z_val[:, args.n_frames_input:], pred_z_val[:, args.n_frames_input:], noise_logvar_z[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                val_loss_output = -logpx_val_output.item()
                val_loss_meter_output.update(val_loss_output)
                val_losses_output.append(val_loss_output)
                val_losses_ma_output.append(val_loss_meter_output.avg)

                xs_dec_z_val = xs_dec_z_val.cpu()
                xs_dec_pred_z_val = xs_dec_pred_z_val.cpu()

                frames_val = []
                for t in range(xs_dec_pred_z_val.shape[1]):
                    xs_t = orig_trajs_val[:, t]                         # Bx64x64
                    xs_dec_z_t = xs_dec_z_val[:, t]                  # Bx64x64
                    xs_dec_pred_z_t = xs_dec_pred_z_val[:, t]      # Bx64x64
                    frame = torch.cat([xs_t, torch.ones(args.vis_n_vids, args.im_ch, args.imsize, 2),
                                       xs_dec_z_t, torch.ones(args.vis_n_vids, args.im_ch, args.imsize, 2),
                                       xs_dec_pred_z_t], dim=-1)
                    gif_frame = vutils.make_grid(frame, nrow=5, padding=8, pad_value=1).permute(1, 2, 0).add(1.).mul(0.5).numpy()
                    frames_val.append((putText(np.concatenate((np.ones((40, gif_frame.shape[1], gif_frame.shape[2])), gif_frame), axis=0), f"time = {t+1}", (8, 30), 0, 1, (0,0,0), 4)*255).astype('uint8'))
                    del frame, gif_frame

                imageio.mimwrite(os.path.join(args.save_path, 'samples', f'val_vis_{itr:06d}.gif'), frames_val, fps=4)
                del xs_dec_z_val, xs_dec_pred_z_val, logpx_val_input, logpx_val_output, frames_val

                # Plot
                plt.plot(np.arange(1, itr+1), losses, '--', c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(1, itr+1), losses_ma, c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(itr//args.vis_step)*args.vis_step + args.vis_step, val_losses_input, '--', c='C1', alpha=0.5, label="val_loss_input")
                plt.plot(np.arange(itr//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_input, c='C1', alpha=0.5, label="val_loss_ma_input")
                plt.plot(np.arange(itr//args.vis_step)*args.vis_step + args.vis_step, val_losses_output, '--', c='C2', alpha=0.5, label="val_loss_output")
                plt.plot(np.arange(itr//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_output, c='C2', alpha=0.5, label="val_loss_ma_output")
                plt.legend()
                plt.yscale("symlog")
                plt.xlabel("Iterations")
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
        'orig_trajs': orig_trajs,
        'samp_trajs': samp_trajs,
        'orig_ts': orig_ts,
        'samp_ts': samp_ts,
    }, ckpt_path)
    log = 'Stored ckpt at {}\n'.format(ckpt_path)
    print(log)
    log_file.write(log)
    log_file.flush()

    log = 'Training complete after {} iters.\n'.format(itr)
    print(log)
    log_file.write(log)
    log_file.flush()
