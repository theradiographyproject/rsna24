import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import zarr

class Dataset(IterableDataset):
    def __init__(self):
        df1 = pd.read_csv("/cbica/home/gangarav/rsna24_raw/train_label_coordinates.csv")
        df1 = df1[df1["condition"] == "Spinal Canal Stenosis"]
        df1 = df1[df1["level"] == "L5/S1"]
        self.condition_coordinates_metadata = df1
        df2 = pd.read_pickle("/cbica/home/gangarav/rsna24/dataframe.pkl")
        self.series_metadata = df2
        

    def convert_pixel_coords_to_action_space(self, d, r, c, series_shape):
        return np.array([
          2*(-0.5 + float(d)/series_shape[0]),
          2*(-0.5 + float(r)/series_shape[1]),
          2*(-0.5 + float(c)/series_shape[2]),
        ])

    def __iter__(self):
        i = 0
        while i < 1000:
            try:
              row = self.condition_coordinates_metadata.sample().iloc[0]
  
              metadata = self.series_metadata[self.series_metadata['series'] == str(row.series_id)].iloc[0]
              selected_dicom = f"/cbica/home/gangarav/RSNA_PROCESSED/dicoms/{row.study_id}/{row.series_id}.zarr"
              zr =  zarr.open(selected_dicom, mode='r')[:]
              zr = (zr - metadata['mean'])/metadata['std']
              spacing = np.array([float(metadata['z_spacing']), float(metadata['y_spacing']), float(metadata['x_spacing'])])
            except:
              print(row.series, "DID NOT WORK")
              continue
            i += 1
            yield zr, spacing, self.convert_pixel_coords_to_action_space(row.instance_number, row.y, row.x, zr.shape)
    

def get_train_valid_loader():
    dataset = Dataset()
  
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1
    )
    return (train_loader, None)

# def get_train_valid_loader(
#     data_dir,
#     batch_size,
#     random_seed,
#     valid_size=0.1,
#     shuffle=True,
#     show_sample=False,
#     num_workers=4,
#     pin_memory=False,
# ):
#     """Train and validation data loaders.

#     If using CUDA, num_workers should be set to 1 and pin_memory to True.

#     Args:
#         data_dir: path directory to the dataset.
#         batch_size: how many samples per batch to load.
#         random_seed: fix seed for reproducibility.
#         valid_size: percentage split of the training set used for
#             the validation set. Should be a float in the range [0, 1].
#             In the paper, this number is set to 0.1.
#         shuffle: whether to shuffle the train/validation indices.
#         show_sample: plot 9x9 sample grid of the dataset.
#         num_workers: number of subprocesses to use when loading the dataset.
#         pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
#             True if using GPU.
#     """
#     error_msg = "[!] valid_size should be in the range [0, 1]."
#     assert (valid_size >= 0) and (valid_size <= 1), error_msg

#     # define transforms
#     normalize = transforms.Normalize((0.1307,), (0.3081,))
#     trans = transforms.Compose([transforms.ToTensor(), normalize])

#     # load dataset
#     dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

#     num_train = len(dataset)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))

#     if shuffle:
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)

#     train_idx, valid_idx = indices[split:], indices[:split]

#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)

#     train_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )

#     valid_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=valid_sampler,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#     )

#     # visualize some images
#     if show_sample:
#         sample_loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=9,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )
#         data_iter = iter(sample_loader)
#         images, labels = data_iter.next()
#         X = images.numpy()
#         X = np.transpose(X, [0, 2, 3, 1])
#         plot_images(X, labels)

#     return (train_loader, valid_loader)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader