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
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import tqdm

from moving_mnist import *

from pathlib import Path
import imageio


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
parser.add_argument('--num_of_samples', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_digits', type=int, default=1)
parser.add_argument('--n_frames_input', type=int, default=10)
parser.add_argument('--n_frames_output', type=int, default=10)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--vis_step', type=int, default=10)
parser.add_argument('--vis_n_vids', type=int, default=50, help="How many videos to visualize")
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu', type=int, default=0)
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


class EncoderRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=16, nhidden=25):
        super(EncoderRNN, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.res_block1 = ResBlock(1, 16, 2, 'downsample')    # 32x32
        self.res_block2 = ResBlock(16, 32, 2, 'downsample')    # 16x16
        self.res_block3 = ResBlock(32, 64, 2, 'downsample')    # 8x8
        self.res_block4 = ResBlock(64, 2, 2, 'downsample')    # 4x4; 2*4*4 = 32 = 2*obs_dim
        self.i2h = nn.Linear(2*obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        bs = x.shape[0]
        x = self.res_block4(self.res_block3(self.res_block2(self.res_block1(x))))
        x = x.view(bs, -1)
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.nhidden)


class Encoder(nn.Module):

    def __init__(self, latent_dim=128, obs_dim=1024, nhidden=256):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.res_block1 = ResBlock(1, 16, 2, 'downsample')    # 32x32
        self.res_block2 = ResBlock(16, 32, 2, 'downsample')    # 16x16
        self.res_block3 = ResBlock(32, 64, 2, 'downsample')    # 8x8
        self.res_block4 = ResBlock(64, 64, 2, 'downsample')    # 4x4; 4*4*64 = 1024 = obs_dim
        self.i2h = nn.Linear(obs_dim, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.res_block4(self.res_block3(self.res_block2(self.res_block1(x))))
        x = x.view(-1, self.obs_dim)
        h = torch.tanh(self.i2h(x))
        out = self.h2o(h)
        return out


class Decoder(nn.Module):

    def __init__(self, latent_dim=128, obs_dim=1024, nhidden=256):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.nhidden = nhidden
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)
        self.res_block1 = ResBlock(64, 64, 2, 'upsample')
        self.res_block2 = ResBlock(64, 32, 2, 'upsample')
        self.res_block3 = ResBlock(32, 16, 2, 'upsample')
        self.res_block4 = ResBlock(16, 1, 2, 'upsample')
        self.tanh = nn.Tanh()

    def forward(self, z):
        bs = z.shape[0]
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(-1, 64, 4, 4)
        out = self.res_block4(self.res_block3(self.res_block2(self.res_block1(out))))
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

# some utilitaries to save visual data (gifs, mp4s ...)
def select_(tensors, dim, index):
    """
    Selects element at position `index` along the provided dimension `dim`.
    """
    return tuple(tensor.select(dim=dim, index=index) for tensor in tensors)


def group_(tensors, padding, dim, pad_value=1):
    """
    Groups tensors along the provided dimension `dim` with some defined padding.
    """
    ref_tensor = tensors[0]
    nb, nc, height, width = ref_tensor.size()
    vpad = pad_value * torch.ones(size=(nb, nc, height, padding))
    tsrs = list()
    for i, tensor in enumerate(tensors):
        if i == 0:
            tsrs.append(tensor)
        else:
            tsrs.append(vpad)
            tsrs.append(tensor)
    group = torch.cat(tensors=tsrs, dim=dim)
    return group


def make_grid_(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Turns the provided batch of images (`tensor` of shape `[batch-size, n-channels, height, width]` ) into an grid of images.

    :param tensor:
    :param nrow:
    :param padding:
    :param normalize:
    :param range:
    :param scale_each:
    :param pad_value:
    :return:
    """
    # tensor is of shape BxCxHxW : [batch-size, n-channels, height, width]
    grid = vutils.make_grid(tensor=tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    array = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return array


def sampler(x):
    while True:
        for e in x:
            yield e


class Writer(object):

    def __init__(self, select_dim, group_dim, group_padding=2, group_pad_value=1, nrows=8, grid_padding=8,
                 grid_pad_value=0, fps=4, dpi=300):
        self.select_dim = select_dim
        self.group_dim = group_dim
        self.group_pad = group_padding
        self.group_pad_value = group_pad_value
        self.nrows = nrows
        self.grid_padding = grid_padding
        self.grid_pad_value = grid_pad_value
        self.fps = fps
        self.dpi = dpi

    def __process(self, inputs):
        outputs = []
        _range = range(inputs[0].size(self.select_dim))
        for t in _range:

            # 
            tensors = select_(tensors=inputs, dim=self.select_dim, index=t)
            group = group_(tensors=tensors, dim=self.group_dim, padding=self.group_pad, pad_value=self.group_pad_value)
            grid = make_grid_(tensor=group, nrow=self.nrows, padding=self.grid_padding, pad_value=self.grid_pad_value)

            # 
            fig, ax = plt.subplots(nrows=1, ncols=1, dpi=self.dpi)
            ax.imshow(grid)
            ax.set_title(f"time: {t-_range[0]}")
            ax.axis("off")

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # draw the renderer
            fig.canvas.draw_idle()
        
            #get the GBA buffer from the figure
            w, h = fig.canvas.get_width_height()
            buf = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8 )
            plt.close()

            buf = buf.reshape([h, w, 3])
            outputs.append(buf)
        return outputs

    def save(self, tensors, uri):
        sequences = self.__process(inputs=tensors)        
        imageio.mimwrite(uri=uri, ims=sequences, fps=self.fps)

    def __call__(self, tensors, uri):
        self.save(tensors=tensors, uri=uri)


if __name__ == '__main__':

    if not os.path.exists(args.save_path):
        args.save_path = os.path.join(os.path.dirname(args.save_path), '{0:%Y%m%d_%H%M%S}_{1}'.format(datetime.datetime.now(), os.path.basename(args.save_path)))
        os.makedirs(args.save_path)
        os.makedirs(os.path.join(args.save_path, 'samples'))

    latent_dim = 128
    obs_dim = 1024
    # rnn_nhidden = 50
    nhidden = 256
    start = 0.
    stop = 1.
    noise_std = .3
    a = 0.
    b = .3
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Data
    print("Loading data")
    dl = moving_mnist_ode_data_loader(args.data_path, num_objects=[args.num_digits], batch_size=args.num_of_samples,
                                        n_frames_input=args.n_frames_input, n_frames_output=args.n_frames_output,
                                        n_workers=8)
    data_loader = sampler(dl)
    data_writer = Writer(select_dim=1, group_dim=-1, group_padding=2, group_pad_value=1, grid_padding=8, grid_pad_value=1, fps=2)

    orig_trajs, samp_trajs, orig_ts, samp_ts = next(data_loader)
    samp_trajs, samp_ts = samp_trajs.to(device), samp_ts.to(device)
    orig_trajs_vis = orig_trajs[:args.vis_n_vids].to(device)

    # Val data
    print("Loading val data")
    orig_trajs_val, _, orig_ts_val, _ = next(data_loader)
    orig_trajs_val = orig_trajs_val[:args.vis_n_vids]
    orig_trajs_val_gpu = orig_trajs_val.clone().to(device)

    # Model
    print("Making models")
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    enc = Encoder(latent_dim, obs_dim, nhidden).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
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
    ts_pos = np.linspace(0., 1., num=args.n_frames_input+args.n_frames_output)
    ts_pos = torch.from_numpy(ts_pos).float().to(device)
    orig_xs = orig_trajs[:args.vis_n_vids]

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

    noise_std_ = torch.zeros(args.batch_size, args.n_frames_input, 1, 64, 64) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    noise_std_z = torch.zeros(args.batch_size, args.n_frames_input, latent_dim) + noise_std
    noise_logvar_z = 2. * torch.log(noise_std_z).to(device)

    try:
        print("Starting training...")
        n_batches = args.num_of_samples//args.batch_size
        print("n_batches", n_batches)

        vid_ids = np.arange(args.num_of_samples)

        for itr in range(1, args.niters + 1):
            total_loss = 0
            np.random.shuffle(vid_ids)

            # TRAIN
            for b in range(n_batches):
                # print("itr", itr)
                optimizer.zero_grad()
                z = enc(samp_trajs[vid_ids[b*args.batch_size:(b+1)*args.batch_size]].view(-1, 1, 64, 64)).view(args.batch_size, -1, latent_dim)
                # forward in time and solve ode for reconstructions
                # print("doing ode")
                pred_z = odeint(func, z.permute(1, 0, 2)[0], samp_ts).permute(1, 0, 2)     # B x T x dim
                # print("decoding after ode")
                pred_x = dec(pred_z)    # BxTx1x64x64

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
                    z = enc(orig_trajs_vis[:args.vis_n_vids].view(-1, 1, 64, 64)).view(args.vis_n_vids, -1, latent_dim)   # BxTxdim
                    xs_dec_z = dec(z)    # BxTx1x64x64
                    pred_z = odeint(func, z.permute(1, 0, 2)[0], ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z = dec(pred_z)    # BxTx1x64x64

                xs_dec_z = xs_dec_z.cpu()
                xs_dec_pred_z = xs_dec_pred_z.cpu()
                
                data_writer.save(tensors=[orig_xs, xs_dec_z, xs_dec_pred_z], uri=Path(args.save_path) / "samples" / "train" / f"vis_{itr:06d}.gif")
                del xs_dec_z, xs_dec_pred_z


                # Sampling from VAL data
                # print("Sampling")
                with torch.no_grad():
                    z_val = enc(orig_trajs_val_gpu.view(-1, 1, 64, 64)).view(args.vis_n_vids, -1, latent_dim)   # BxTxdim
                    xs_dec_z_val = dec(z_val)    # BxTx1x64x64
                    pred_z_val = odeint(func, z_val.permute(1, 0, 2)[0], ts_pos).permute(1, 0, 2) # BxTxdim
                    xs_dec_pred_z_val = dec(pred_z_val)    # BxTx1x64x64

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

                data_writer.save(tensors=[orig_trajs_val, xs_dec_z_val, xs_dec_pred_z_val], uri=Path(args.save_path) / "samples" / "valid" / f"vis_{itr:06d}.gif")
                del xs_dec_z_val, xs_dec_pred_z_val

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
                plt.savefig(os.path.join(args.save_path, "loss.png"), bbox_inches='tight', pad_inches=0.1, dpi=300)
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
