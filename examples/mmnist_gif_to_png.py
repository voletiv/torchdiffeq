import imageio
import matplotlib.pyplot as plt
import numpy as np

gif_file = '/home/voletiv/EXPERIMENTS/ode/20190819_155734_ODE_PLEASE_WORK_AGAIN/samples/train_vis_004900.gif'
h, w = 10, 5
vid_r, vid_c = 4, 1

t, l = -8*(h - vid_r) - 64*(h - vid_r), 8*(vid_c + 1) + 64*3*(vid_c) + 4*(vid_c)
frames = np.array(imageio.mimread(gif_file))
my_frames = np.concatenate([frames[:, t-2:t+64+2, l:l+64], frames[:, t:t+64+2, l+64+2+64+2:l+64+2+64+2+64]], axis=1)
m = np.hstack([np.hstack([my_frames[i], 255*np.ones((my_frames[i].shape[0], 2))]) for i in range(len(my_frames))])
imageio.imwrite('/home/voletiv/aa.png', m)



gif_file = '/home/voletiv/EXPERIMENTS/ode/20190908_221711_ODE_RNNEncSmall_10xvids_smallZ_CORRECTval_nVids10000_nRes4_8_16_32_32_hid128_z64_skip0_bs128_lr0.0001_seed0/samples/val_vis_001900.gif'
h, w = 5, 10
vid_r, vid_c = 2, 2

t, l = -8*(h - vid_r) - 64*(h - vid_r), 8*(vid_c + 1) + 64*2*(vid_c) + 2*(vid_c)
frames = np.array(imageio.mimread(gif_file))
my_frames = np.concatenate([frames[:, t-2:t+64+2, l:l+64], frames[:, t:t+64+2, l+64+2:l+64+2+64]], axis=1)
m = np.hstack([np.hstack([my_frames[i], 255*np.ones((my_frames[i].shape[0], 2))]) for i in range(len(my_frames))])
imageio.imwrite('/home/voletiv/aa.png', m)



gif_file = '/home/voletiv/EXPERIMENTS/ode/20190908_224438_ODE_RNNEnc_MMNIST_2DIGITS_nVids10000_nRes4_8_16_32_32_hid128_z64_skip0_bs128_lr0.0001_seed0/samples/val_vis_002000.gif'
h, w = 5, 10
vid_r, vid_c = 2, 1

t, l = -8*(h - vid_r) - 64*(h - vid_r), 8*(vid_c + 1) + 64*2*(vid_c) + 2*(vid_c)
frames = np.array(imageio.mimread(gif_file))
my_frames = np.concatenate([frames[:, t-2:t+64+2, l:l+64], frames[:, t:t+64+2, l+64+2:l+64+2+64]], axis=1)
m = np.hstack([np.hstack([my_frames[i], 255*np.ones((my_frames[i].shape[0], 2))]) for i in range(len(my_frames))])
imageio.imwrite('/home/voletiv/aa.png', m)

