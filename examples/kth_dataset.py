import glob
import h5py
import imageio
import numpy as np
import os
import torch

from tqdm import tqdm

# from cv2 import resize
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Resize, ToTensor

# import sys; del sys.modules['kth_dataset']; from kth_dataset import *
# ds = KTH_Dataset('/home/voletiv/aa.h5'); dl = torch.utils.data.DataLoader(ds, batch_size=7, shuffle=False, collate_fn=lambda batch: ds.kth_collate_fn(batch)); di = iter(dl); a = next(di)


class KTH_Dataset(Dataset):

    def __init__(self, h5_path, val=False, classes=None,
                 n_frames_cond=10, n_frames_pred=10, n_frames_future=30,
                 time_start=0, time_stop_cond=0.5):
        super().__init__()
        self.h5_path = h5_path
        self.val = val
        self.n_frames_cond = n_frames_cond
        self.n_frames_pred = n_frames_pred
        self.n_frames_future = n_frames_future
        self.n_frames_total = n_frames_cond + n_frames_pred + n_frames_future
        self.time_start = time_start
        self.time_stop_cond = time_stop_cond

        with h5py.File(self.h5_path, 'r') as h5_file:
            self.kth_classes = [l.decode() for l in h5_file["labels"][:]]
            lengths = h5_file['lengths'][:]

        if classes is None:
            self.labels = self.kth_classes
        else:
            self.labels = classes

        self.lengths = lengths[[self.kth_classes.index(c) for c in self.labels]]
        if self.val:
            self.offsets = np.array([l - l//10 for l in self.lengths])
            self.lengths = np.array([l//10 for l in self.lengths])
        else:
            self.offsets = np.zeros(len(self.lengths), dtype=int)
            self.lengths = np.array([l - l//10 for l in self.lengths])

        self.len = self.lengths.sum()
        self.len_bins = np.cumsum(self.lengths)

    def kth_collate_fn(self, batch):
        idxs = batch

        sorted_idxs, inv_idxs = np.unique(idxs, return_inverse=True)

        label_idxs = np.digitize(sorted_idxs, self.len_bins)
        idxs_in_labels = np.array([self.offsets[label_idxs[i]] + (sorted_idxs[i] % self.lengths[label_idxs[i]]) for i in range(len(sorted_idxs))])

        unique_label_idxs, inv_label_idxs = np.unique(label_idxs, return_inverse=True)
        vids = []
        vid_lengths = []
        with h5py.File(self.h5_path, 'r') as h5_file:
            for l, label_idx in enumerate(unique_label_idxs):
                idxs_in_batch = np.where(inv_label_idxs == l)[0]
                idxs_in_label = idxs_in_labels[idxs_in_batch]
                unique_idxs_in_label, inv_idxs_in_label = np.unique(idxs_in_label, return_inverse=True)
                vids += (h5_file[self.labels[label_idx]]['vids'][unique_idxs_in_label.tolist()][inv_idxs_in_label]).tolist()
                vid_lengths += (h5_file[self.labels[label_idx]]['len'][unique_idxs_in_label.tolist()][inv_idxs_in_label]).tolist()

        label_idxs = label_idxs[inv_idxs]

        # Randomly sample n_frames_total frames, centre crop to 120
        rand_idx = [np.random.randint(vid_lengths[i] - self.n_frames_total) for i in range(len(vid_lengths))]
        vids = np.array([vids[i].reshape(-1, 120, 160)[rand_idx[i]:rand_idx[i] + self.n_frames_total, :, 20:140] for i in inv_idxs])

        # Resize to 64, value range [0., 1.]
        vids = torch.stack([torch.stack([ToTensor()(Resize(64)(ToPILImage()(frame[:, :, np.newaxis]))) for frame in vid]) for vid in vids]).mul(2.).sub(1.)

        # Return vids_cond, vids_pred, vids_future, label_idxs
        return vids[:, :self.n_frames_cond], vids[:, self.n_frames_cond:self.n_frames_cond+self.n_frames_pred], vids[:, self.n_frames_cond+self.n_frames_pred:], label_idxs

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.len


def make_kth_h5_dataset(h5_path, kth_path):
    """kth_path: Path to directory containing all KTH sub-directories"""
    kth_classes = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
    kth_classes_ascii = [l.encode("ascii", "ignore") for l in kth_classes]
    spl_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
    with h5py.File(h5_path, 'w') as h5_file:
        # Labels
        h5_file.create_dataset("labels", (len(kth_classes),), 'S' + str(max([len(i) for i in kth_classes])),
                                data=kth_classes_ascii)
        for label in tqdm(kth_classes):
            label_vid_files = sorted(glob.glob(os.path.join(kth_path, label, "*.avi")))
            # Create group
            h5_file.create_group(label)
            # Create dataset of lengths
            h5_file[label].create_dataset("len", (len(label_vid_files),), dtype=int)
            # # Create group of vids
            # h5_file[label].create_group("vids")
            # Create dataset of vids
            h5_file[label].create_dataset("vids", (len(label_vid_files),), dtype=spl_dtype)
            # For each video file
            for i, vid_file in tqdm(enumerate(label_vid_files), total=len(label_vid_files)):
                # Read video
                vid = imageio.mimread(vid_file)
                vid = [f[:, :, 0] for f in vid]
                # Save length of video
                h5_file[label]["len"][i] = len(vid)
                # Save video
                h5_file[label]["vids"][i] = np.array(vid).reshape(-1)   # To read video: a = h5_file[label]["vids"][idx]; [i.reshape(120, 160, -1) for i in a]
                # # Save buffer string of GIF of full video
                # buffer_ = io.BytesIO()
                # imageio.mimsave(buffer_, vid, format='gif', fps=10)
                # buffer_.seek(0)
                # vid = np.frombuffer(buffer_.read(), dtype='uint8')    # To read video: np.array(imageio.mimread(io.BytesIO(vid_bytes)))
                # h5_file[label]["vids"][i] = vid
                # # Create dataset of video frames, with key as string of index
                # h5_file[label]["vids"].create_dataset(str(i), (len(vid),), dtype=spl_dtype)
                # # Save each frame as IO buffer
                # for f, frame in enumerate(vid):
                #     buffer_ = io.BytesIO()
                #     Image.fromarray(vid[0]).save(buffer_, format='png')
                #     h5_file[label]["vids"][str(i)][f] = np.frombuffer(buffer_.getvalue(), dtype='uint8')
        h5_file.create_dataset("lengths", (len(kth_classes),), dtype=int,
                                data=[len(h5_file[label]["len"]) for label in [l.decode() for l in h5_file["labels"]]])

