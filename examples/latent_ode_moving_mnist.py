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

# python latent_ode_moving_mnist.py --save_path /home/voletivi/scratch/ode/ODE_RNNEnc --n_vids 10000 --batch_size 128 --n_res_blocks 4 --res_ch 8 16 32 32 --nhidden 128 --latent_dim 64 --vis_step 100 --vis_n_vids 50 --num_workers 8 --n_epochs 1001 --seed 0

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
parser.add_argument('--n_vids', type=int, default=10000)
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


# class Encoder(nn.Module):

#     def __init__(self, latent_dim=128, obs_dim=1024, nhidden=256, skip_level=1):
#         super(Encoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.obs_dim = obs_dim
#         self.nhidden = nhidden
#         self.skip_level = skip_level
#         self.res_block1 = ResBlock(1, 16, 2, 'downsample')      # 32x32 <-- 64x64
#         self.res_block2 = ResBlock(16, 32, 2, 'downsample')     # 16x16
#         self.res_block3 = ResBlock(32, 64, 2, 'downsample')     # 8x8
#         self.res_block4 = ResBlock(64, 64, 2, 'downsample')     # 4x4; 4*4*64 = 1024 = obs_dim
#         self.i2h = nn.Linear(obs_dim, nhidden)
#         self.h2o = nn.Linear(nhidden, latent_dim)

#     def forward(self, x):
#         # bs = x.shape[0]
#         x1 = self.res_block1(x)     # 32x32
#         x2 = self.res_block2(x1)    # 16x16
#         x3 = self.res_block3(x2)    # 8x8
#         x4 = self.res_block4(x3)    # 4x4
#         x = x4.view(-1, self.obs_dim)
#         h = torch.tanh(self.i2h(x))
#         out = self.h2o(h)
#         if self.skip_level == 1:
#             feats = x1
#         elif self.skip_level == 2:
#             feats = x2
#         elif self.skip_level == 3:
#             feats = x3
#         elif self.skip_level == 4:
#             feats = x4
#         return out, feats


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
            f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.path.basename(args.save_path)}_nVids{args.n_vids}_nRes{args.n_res_blocks}_')
        for i in range(args.n_res_blocks):
            args.save_path += f'{args.res_ch[i]}_'
        args.save_path += f'hid{args.nhidden}_z{args.latent_dim}_skip{args.skip_level}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}'
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'samples'))
        os.makedirs(os.path.join(args.save_path, 'checkpoints'))
    else:
        ckpt_path = os.path.join(args.save_path, 'checkpoints', 'ckpt.pth')
        if os.path.exists(ckpt_path):
            print("Loading model ckpt from", ckpt_path)
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            enc.load_state_dict(checkpoint['enc_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

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

    # Val data
    print("Loading val data")
    orig_trajs_val, _, orig_ts_val, _ = next(iter(dl))
    orig_trajs_val = orig_trajs_val[:args.vis_n_vids]
    orig_trajs_val_gpu = orig_trajs_val.clone().to(device)

    # Model
    print("Making models")
    func = LatentODEfunc(args.latent_dim, args.nhidden).to(device)
    enc = EncoderRNN(args.n_res_blocks, args.res_ch, args.nhidden, args.latent_dim, args.skip_level, args.imsize, args.im_ch).to(device)
    # enc = Encoder(args.latent_dim, args.obs_dim, args.nhidden, args.skip_level).to(device)
    dec = Decoder(args.latent_dim, args.nhidden, args.n_res_blocks, args.res_ch[::-1], args.skip_level, args.imsize, args.im_ch).to(device)
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

    log_file_name = os.path.join(args.save_path, 'log.txt')
    log_file = open(log_file_name, "wt")
    log_file.write(str(args))
    log_file.flush()

    noise_std_ = torch.zeros(args.batch_size, args.n_frames_input, args.im_ch, args.imsize, args.imsize) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    noise_std_z = torch.zeros(args.batch_size, args.n_frames_input, args.latent_dim) + noise_std
    noise_logvar_z = 2. * torch.log(noise_std_z).to(device)

    try:
        print("Starting training...")
        n_batches = args.n_vids//args.batch_size
        print("n_batches", n_batches)

        vid_ids = np.arange(args.n_vids)

        for epoch in range(1, args.n_epochs + 1):
            total_loss = 0
            np.random.shuffle(vid_ids)

            # TRAIN
            for b in range(n_batches):
                # print("epoch", epoch)
                optimizer.zero_grad()
                # z, feats = enc(samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size]].view(-1, args.im_ch, args.imsize, args.imsize))     # B*Txdim
                # z = z.view(args.batch_size, -1, args.latent_dim)    # BxTxdim
                # feats = feats.view(args.batch_size, -1, *feats.shape[1:])[:, 0:1].expand(args.batch_size, feats.shape[0]//args.batch_size, *feats.shape[1:]).contiguous().view(-1, *feats.shape[1:])
                # backward in time to infer q(z_0)
                h = enc.initHidden(args.batch_size).to(device)
                for t in reversed(range(samp_trajs.size(1))):
                    obs = samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size], t, :]
                    out, h, feats = enc.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions
                # print("doing ode")
                # pred_z = odeint(func, z.permute(1, 0, 2)[0], samp_ts).permute(1, 0, 2)     # BxTxdim
                pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)     # BxTxdim
                # print("decoding after ode")
                pred_x = dec(pred_z, feats)    # BxTx1x64x64

                # compute loss
                # print("computing loss")
                # import pdb; pdb.set_trace()
                logpx = log_normal_pdf(samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size]], pred_x, noise_logvar).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                # logpx += log_normal_pdf(z, pred_z, noise_logvar_z).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                # loss = -logpx
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

            loss_meter.update(total_loss/n_batches)
            losses.append(total_loss/n_batches)
            losses_ma.append(loss_meter.avg)

            log = 'Epoch: {}, running avg loss: {:.4f}\n'.format(epoch, loss_meter.avg)
            print(args.save_path)
            print(log)
            log_file.write(log)
            log_file.flush()

            # SAVE MODEL
            if epoch % args.model_save_step == 0:
                ckpt_path = os.path.join(args.save_path, 'checkpoints', f'ckpt_{epoch:05d}.pth')
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'enc_state_dict': enc.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, ckpt_path)
                log = 'Stored ckpt at {}\n'.format(ckpt_path)
                print(log)
                # log_file.write(log)
                # log_file.flush()

            # VISUALIZE
            if epoch % args.vis_step == 0:

                # Sampling from TRAIN data
                # print("Sampling")
                with torch.no_grad():
                    # z, feats = enc(orig_trajs_vis_gpu.view(-1, args.im_ch, args.imsize, args.imsize))   # B*Txdim
                    # z = z.view(args.vis_n_vids, -1, args.latent_dim)    # BxTxdim
                    # feats = feats.view(args.vis_n_vids, -1, *feats.shape[1:])[:, 0:1].expand(args.vis_n_vids, feats.shape[0]//args.vis_n_vids, *feats.shape[1:]).contiguous().view(-1, *feats.shape[1:])
                    # xs_dec_z = dec(z, feats)    # BxTx1x64x64
                    h = enc.initHidden(args.vis_n_vids).to(device)
                    for t in reversed(range(orig_trajs_vis_gpu.size(1)//2)):
                        obs = orig_trajs_vis_gpu[:, t, :]
                        out, h, feats = enc.forward(obs, h)
                    qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
                    epsilon = torch.randn(qz0_mean.size()).to(device)
                    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                    pred_z = odeint(func, z0, ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z = dec(pred_z, feats)    # BxTx1x64x64

                # xs_dec_z = xs_dec_z.cpu()
                xs_dec_pred_z = xs_dec_pred_z.cpu()

                frames = []
                for t in range(xs_dec_pred_z.shape[1]):
                    xs_t = orig_trajs_vis[:, t]                        # Bx1x64x64
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
                # import pdb; pdb.set_trace()
                with torch.no_grad():
                    # z_val, feats_val = enc(orig_trajs_val_gpu.view(-1, args.im_ch, args.imsize, args.imsize))   # BxTxdim
                    # z_val = z_val.view(args.vis_n_vids, -1, args.latent_dim)    # BxTxdim
                    # feats_val = feats_val.view(args.vis_n_vids, -1, *feats_val.shape[1:])[:, 0:1].expand(args.vis_n_vids, feats_val.shape[0]//args.vis_n_vids, *feats_val.shape[1:]).contiguous().view(-1, *feats_val.shape[1:])
                    # xs_dec_z_val = dec(z_val, feats_val)    # BxTx1x64x64
                    h = enc.initHidden(args.vis_n_vids).to(device)
                    for t in reversed(range(orig_trajs_val_gpu.size(1)//2)):
                        obs = orig_trajs_val_gpu[:, t, :]
                        out, h, feats = enc.forward(obs, h)
                    qz0_mean_val, qz0_logvar_val = out[:, :args.latent_dim], out[:, args.latent_dim:]
                    epsilon = torch.randn(qz0_mean.size()).to(device)
                    z0_val = epsilon * torch.exp(.5 * qz0_logvar_val) + qz0_mean_val

                    pred_z_val = odeint(func, z0_val, ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z_val = dec(pred_z_val, feats)  # BxTx1x64x64

                logpx_val_input = log_normal_pdf(orig_trajs_val_gpu[:, :args.n_frames_input], xs_dec_pred_z_val[:, :args.n_frames_input], noise_logvar[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                # logpx_val_input += log_normal_pdf(z_val[:, :args.n_frames_input], pred_z_val[:, :args.n_frames_input], noise_logvar_z[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                
                pz0_mean_val = pz0_logvar_val = torch.zeros(z0_val.size()).to(device)
                analytic_kl_val = normal_kl(qz0_mean_val, qz0_logvar_val,
                                            pz0_mean_val, pz0_logvar_val).sum(-1)
                val_loss_input = torch.mean(-logpx_val_input + analytic_kl_val, dim=0).item()

                val_loss_meter_input.update(val_loss_input)
                val_losses_input.append(val_loss_input)
                val_losses_ma_input.append(val_loss_meter_input.avg)

                logpx_val_output = log_normal_pdf(orig_trajs_val_gpu[:, args.n_frames_input:], xs_dec_pred_z_val[:, args.n_frames_input:], noise_logvar[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                # logpx_val_output += log_normal_pdf(z_val[:, args.n_frames_input:], pred_z_val[:, args.n_frames_input:], noise_logvar_z[:args.vis_n_vids]).sum(-1).sum(-1).sum(-1).sum(-1).mean()
                val_loss_output = -logpx_val_output.item()
                val_loss_meter_output.update(val_loss_output)
                val_losses_output.append(val_loss_output)
                val_losses_ma_output.append(val_loss_meter_output.avg)

                # xs_dec_z_val = xs_dec_z_val.cpu()
                xs_dec_pred_z_val = xs_dec_pred_z_val.cpu()

                frames_val = []
                for t in range(xs_dec_pred_z_val.shape[1]):
                    xs_t = orig_trajs_val[:, t]                         # Bx64x64
                    # xs_dec_z_t = xs_dec_z_val[:, t]                  # Bx64x64
                    xs_dec_pred_z_t = xs_dec_pred_z_val[:, t]      # Bx64x64
                    frame = torch.cat([xs_t, torch.ones(args.vis_n_vids, args.im_ch, args.imsize, 2),
                                       # xs_dec_z_t, torch.ones(args.vis_n_vids, args.im_ch, args.imsize, 2),
                                       xs_dec_pred_z_t], dim=-1)
                    gif_frame = vutils.make_grid(frame, nrow=10, padding=8, pad_value=1).permute(1, 2, 0).add(1.).mul(0.5).numpy()
                    frames_val.append((putText(np.concatenate((np.ones((40, gif_frame.shape[1], gif_frame.shape[2])), gif_frame), axis=0), f"time = {t+1}", (8, 30), 0, 1, (0,0,0), 4)*255).astype('uint8'))
                    del frame, gif_frame

                imageio.mimwrite(os.path.join(args.save_path, 'samples', f'val_vis_{epoch:06d}.gif'), frames_val, fps=4)
                del xs_dec_pred_z_val, val_loss_input, val_loss_output, frames_val

                # Plot
                plt.plot(np.arange(1, epoch+1), losses, '--', c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(1, epoch+1), losses_ma, c='C0', alpha=0.7, label="loss")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_input, '--', c='C1', alpha=0.5, label="val_loss_input")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_input, c='C1', alpha=0.5, label="val_loss_ma_input")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_output, '--', c='C2', alpha=0.5, label="val_loss_output")
                plt.plot(np.arange(epoch//args.vis_step)*args.vis_step + args.vis_step, val_losses_ma_output, c='C2', alpha=0.5, label="val_loss_ma_output")
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

    ckpt_path = os.path.join(args.save_path, 'checkpoints', f'ckpt.pth')
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

    log = 'Training complete after {} epochs.\n'.format(epoch)
    print(log)
    log_file.write(log)
    log_file.flush()
