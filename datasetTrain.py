import marimo as mo
from einops import rearrange, repeat, reduce, pack, unpack
from itertools import product
import pydicom
import os
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
import random 
import zarr
import pandas as pd
import math
import lightning as L

from config import raw_data_path, RSNA_2024_competition, LSD_dataset, RSNA_processed, LSD_processed

def get_prism_shape_and_token_count(min_unmasked_token=16, max_unmasked_tokens=48, num_masked_tokens=16, patch_size=(1, 16, 16)):
    r = random.randint(6, 8)
    c = random.randint(6, 8)
    d = random.randint(1, 3)

    max_unmasked_tokens= min(max_unmasked_tokens, ((2**d)/patch_size[0]) * ((2**r)/patch_size[1]) * ((2**c)/patch_size[2]) - num_masked_tokens)
    N = random.randint(min_unmasked_token, max_unmasked_tokens)
    M = num_masked_tokens

    return 2**d, 2**r, 2**c, N, M

def get_frame_indices_for_series_shape_and_slice_shape(series_shape, slice_shape):
    slice_indices = []
    for axis, size in enumerate(series_shape):
        if size < slice_shape[axis]:
            return None
        else:
            index = random.randint(0, size - slice_shape[axis])
            slice_indices.append([index, index + slice_shape[axis]])
    return slice_indices

def get_zarr_reference(series_path):
    return zarr.open(series_path, mode='r')

def get_frames_from_zarr_reference(zarr, series_mean, series_std, slice_list=None):
    if slice_list is not None:
        slices = tuple(slice(start, end) for start, end in slice_list)
        z = zarr[slices]
    else:
        z = zarr[:]

    z = (z - series_mean)/series_std

    return np.expand_dims(z, axis=0)

def enough_dims_in_series_shape(series_shape, slice_shape):
    for axis, size in enumerate(series_shape):
        if size < slice_shape[axis]:
            return False
    return True

def get_source_base_path(source):
    if source == "RSNA2024":
        return RSNA_processed
    elif source == "LSD":
        return LSD_processed

def get_ipp(metadata_path, study, series, idx):
    metadata_path = f"{metadata_path}/dataset_slice_metadata.zarr"
    root = zarr.open(metadata_path, mode='r')
    return root[study][series]['ipp'][idx]

def get_iop(metadata_path, study, series, idx):
    metadata_path = f"{metadata_path}/dataset_slice_metadata.zarr"
    root = zarr.open(metadata_path, mode='r')
    return root[study][series]['iop'][idx]

def get_physical_coordinates(ipp, iop, pixel_spacing, row, col):
    # Row and column vectors derived from IOP
    row_vector = iop[1] * pixel_spacing[1]  # Multiply each row component by the row pixel spacing
    col_vector = iop[0] * pixel_spacing[0]  # Multiply each column component by the column pixel spacing

    # Convert (row, col) to physical coordinates (x, y, z)
    physical_coords = ipp + row * row_vector + col * col_vector
    return physical_coords

def get_physical_coordinates_from_row(df_row, coords):
    base_path = get_source_base_path(df_row["source"])
    idx, row, col = coords
    ipp = get_ipp(base_path, df_row["study"], df_row["series"], idx)
    iop = get_iop(base_path, df_row["study"], df_row["series"], idx)
    return get_physical_coordinates(ipp, iop, [df_row["x_spacing"], df_row["y_spacing"]], row, col)

def categorize_patches(pd, pw, ph, N, M):
    # Create a grid of indices
    indices = np.array(list(product(range(pd), range(pw), range(ph))))

    # Shuffle the indices randomly
    np.random.shuffle(indices)

    # Split indices into non-masked and masked
    nonmask_indices = indices[:N]
    mask_indices = indices[N:N+M]
    skipped = indices[N+M:]

    # Extract specific axes for non-masked indices
    nm_idx_z, nm_idx_x, nm_idx_y = nonmask_indices[:, 0], nonmask_indices[:, 1], nonmask_indices[:, 2]

    # Extract specific axes for masked indices
    m_idx_z, m_idx_x, m_idx_y = mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]

    # Extract specific axes for skipped indices (for debugging pruposes)
    sk_idx_z, sk_idx_x, sk_idx_y = skipped[:, 0], skipped[:, 1], skipped[:, 2]

    return (m_idx_z, m_idx_x, m_idx_y), (nm_idx_z, nm_idx_x, nm_idx_y), (sk_idx_z, sk_idx_x, sk_idx_y), mask_indices, nonmask_indices, skipped

def patchify(prism, offset, patch_size, voxel_spacing, M, N):
    patches = rearrange(prism, '1 (d p1) (r p2) (c p3) -> d r c p1 p2 p3', p1 = patch_size[0], p2 = patch_size[1], p3 = patch_size[2])

    pd, pr, pc = patches.shape[:3]

    m_id, nm_id, sk_id, mask_indices, nonmask_indices, skipped_indices = categorize_patches(pd, pr, pc, N, M)

    m_patch = patches[m_id[0], m_id[1], m_id[2], :, :, :]
    nm_patch = patches[nm_id[0], nm_id[1], nm_id[2], :, :, :]
    sk_patch = patches[sk_id[0], sk_id[1], sk_id[2], :, :, :]

    mask_indices_pt_coords = (mask_indices - np.array([(pd-1)/2, (pr-1)/2, (pc-1)/2])) * patch_size + offset
    mask_indices_pt_coords *= voxel_spacing
    nonmask_indices_pt_coords = (nonmask_indices - np.array([(pd-1)/2, (pr-1)/2, (pc-1)/2])) * patch_size + offset
    nonmask_indices_pt_coords *= voxel_spacing
    skipped_indices_pt_coords = (skipped_indices - np.array([(pd-1)/2, (pr-1)/2, (pc-1)/2])) * patch_size + offset
    skipped_indices_pt_coords *= voxel_spacing

    return m_patch, nm_patch, sk_patch, mask_indices, nonmask_indices, skipped_indices, mask_indices_pt_coords, nonmask_indices_pt_coords, skipped_indices_pt_coords

def load_prism_from_row(row, prism_indices, patch_size, M, N, origin=None):
    if M==0 and N==0:
        # return empty versions of the normal return
        empty_array = np.empty((0, patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
        empty_coords = np.zeros((0, 3), dtype=np.float32)  # Assuming 3D coordinates
        return (empty_array, empty_array, empty_array, 
                empty_coords, empty_coords, empty_coords, 
                empty_array, empty_array, empty_array, 
                empty_coords, empty_coords, empty_coords)

    base_path = get_source_base_path(row["source"])
    series_path = f"{base_path}/dicoms/{row['study']}/{row['series']}.zarr"
    zarr_ref = get_zarr_reference(series_path)
    prism = get_frames_from_zarr_reference(zarr_ref, row["mean"], row["std"], prism_indices)
    midpoint_arr_coords = np.array([np.mean(i_pairs) for i_pairs in prism_indices]) - [0.5, 0.5, 0.5]
    midpoint_patient_coords = get_physical_coordinates_from_row(row, np.floor(midpoint_arr_coords).astype(int))

    offset = [0, 0, 0] if origin is None else midpoint_arr_coords - origin

    m_patch, nm_patch, sk_patch, \
    mask_indices, nonmask_indices, skipped_indices, \
    mask_indices_pt_coords, nonmask_indices_pt_coords, skipped_indices_pt_coords \
    = patchify(prism, offset, patch_size, [row['z_spacing'], row['x_spacing'], row['y_spacing']], M, N)


    return m_patch, nm_patch, sk_patch, mask_indices, nonmask_indices, skipped_indices, prism, midpoint_arr_coords, midpoint_patient_coords, mask_indices_pt_coords, nonmask_indices_pt_coords, skipped_indices_pt_coords

class TrainDataset(IterableDataset):
    def __init__(self, num_batches, mini_batch_size, n_workers, patch_size, support_prism_prob, compare_different_series_prob):
        self.num_batches = int(num_batches/n_workers)
        ### minibatchsize, support_prism, different_series
        self.mini_batch_size = mini_batch_size

        df1 = pd.read_pickle('dataframe.pkl')
        train_df1 = df1[df1['error'].apply(lambda x: x == [])]

        # df2 = pd.read_pickle('lsd_dataframe.pkl')
        # train_df2 = df2[df2['error'].apply(lambda x: x == [])]

        # train_df = pd.concat([train_df1, train_df2], ignore_index=True)

        self.series_metadata = train_df1[:100]

        self.compare_different_series_prob = compare_different_series_prob
        self.support_prism_prob = support_prism_prob

        self.max_unmasked = 48
        self.min_unmasked = 16
        self.num_masked = 16
        self.patch_size = patch_size


    def __iter__(self):
        for i_num in range(self.num_batches):
            pD, pR, pC, pN, pM = get_prism_shape_and_token_count(
                min_unmasked_token=self.min_unmasked,
                max_unmasked_tokens=self.max_unmasked,
                num_masked_tokens=self.num_masked,
                patch_size=self.patch_size
            )

            support_prism = random.random() < self.support_prism_prob

            if support_prism:
                sD, sR, sC, sN, _ = get_prism_shape_and_token_count(
                    min_unmasked_token=0,
                    max_unmasked_tokens=self.max_unmasked-pN,
                    num_masked_tokens=0,
                    patch_size=self.patch_size
                )
            else:
                sD, sR, sC, sN = 0, 0, 0, 0

            batch_multiplier = int(self.max_unmasked/(pN + sN))
            batch = []

            for _ in range(self.mini_batch_size * batch_multiplier):
                compare_different_series = random.random() < self.compare_different_series_prob

                for tries in range(100):
                    row = self.series_metadata.sample().iloc[0]

                    if not enough_dims_in_series_shape(row['shape'], (pD, pR, pC)):
                        print()
                        print("p1_shape", i_num, tries, pD, pR, pC, row['study'], row['series'], row['shape'])
                        print()
                        continue
                    if support_prism and not enough_dims_in_series_shape(row['shape'], (sD, sR, sC)):
                        print()
                        print("s1_shape", i_num, tries, sD, sR, sC, row['study'], row['series'], row['shape'])
                        print()
                        continue

                    if compare_different_series:
                        study_rows = self.series_metadata[(self.series_metadata['study'] == row['study']) & (self.series_metadata['series'] != row['series'])]
                        if study_rows.empty:
                            print()
                            print("no series", i_num, tries, row['study'], row['series'])
                            print()
                            continue
                        cmp_row = study_rows.sample().iloc[0]
                    else:
                        cmp_row = row

                    if not enough_dims_in_series_shape(cmp_row['shape'], (pD, pR, pC)):
                        print()
                        print("p2_shape", i_num, tries, pD, pR, pC, cmp_row['study'], cmp_row['series'], row['shape'])
                        print()
                        continue
                    if support_prism and not enough_dims_in_series_shape(cmp_row['shape'], (sD, sR, sC)):
                        print()
                        print("s2_shape", i_num, tries, sD, sR, sC, cmp_row['study'], cmp_row['series'], row['shape'])
                        print()
                        continue

                    break
                else:
                    print("we were going to be stuck in a while loop")
                    break

                p_1_slices = get_frame_indices_for_series_shape_and_slice_shape(row['shape'], (pD, pR, pC))
                p_1_m_patch, \
                p_1_nm_patch, \
                p_1_sk_patch, \
                p_1_mask_indices, \
                p_1_nonmask_indices, \
                p_1_skipped_indices, \
                p_1_prism, \
                p_1_midpoint_arr_coords, \
                p_1_midpoint_patient_coords, \
                p_1_mask_indices_pt_coords, \
                p_1_nonmask_indices_pt_coords, \
                p_1_skipped_indices_pt_coords \
                = load_prism_from_row(row=row, prism_indices=p_1_slices, patch_size=self.patch_size, M=pM, N=pN)

                s_1_slices = get_frame_indices_for_series_shape_and_slice_shape(row['shape'], (sD, sR, sC))
                s_1_m_patch, \
                s_1_nm_patch, \
                s_1_sk_patch, \
                s_1_mask_indices, \
                s_1_nonmask_indices, \
                s_1_skipped_indices, \
                s_1_prism, \
                s_1_midpoint_arr_coords, \
                s_1_midpoint_patient_coords, \
                s_1_mask_indices_pt_coords, \
                s_1_nonmask_indices_pt_coords, \
                s_1_skipped_indices_pt_coords \
                = load_prism_from_row(row=row, prism_indices=s_1_slices, patch_size=self.patch_size, M=0, N=sN, origin=p_1_midpoint_arr_coords)

                p_2_slices = get_frame_indices_for_series_shape_and_slice_shape(cmp_row['shape'], (pD, pR, pC))
                p_2_m_patch, \
                p_2_nm_patch, \
                p_2_sk_patch, \
                p_2_mask_indices, \
                p_2_nonmask_indices, \
                p_2_skipped_indices, \
                p_2_prism, \
                p_2_midpoint_arr_coords, \
                p_2_midpoint_patient_coords, \
                p_2_mask_indices_pt_coords, \
                p_2_nonmask_indices_pt_coords, \
                p_2_skipped_indices_pt_coords \
                = load_prism_from_row(row=cmp_row, prism_indices=p_2_slices, patch_size=self.patch_size, M=pM, N=pN)

                s_2_slices = get_frame_indices_for_series_shape_and_slice_shape(cmp_row['shape'], (sD, sR, sC))
                s_2_m_patch, \
                s_2_nm_patch, \
                s_2_sk_patch, \
                s_2_mask_indices, \
                s_2_nonmask_indices, \
                s_2_skipped_indices, \
                s_2_prism, \
                s_2_midpoint_arr_coords, \
                s_2_midpoint_patient_coords, \
                s_2_mask_indices_pt_coords, \
                s_2_nonmask_indices_pt_coords, \
                s_2_skipped_indices_pt_coords \
                = load_prism_from_row(row=cmp_row, prism_indices=s_2_slices, patch_size=self.patch_size, M=0, N=sN, origin=p_2_midpoint_arr_coords)

                comparison_vector = p_1_midpoint_patient_coords - p_2_midpoint_patient_coords

                # print(p_1_nm_patch.shape, s_1_nm_patch.shape, p_2_nm_patch.shape, s_2_nm_patch.shape, p_1_nm_patch.dtype, s_1_nm_patch.dtype, p_2_nm_patch.dtype, s_2_nm_patch.dtype)

                non_masked_patches_1 = np.concatenate((p_1_nm_patch, s_1_nm_patch), axis=0)
                non_masked_patches_2 = np.concatenate((p_2_nm_patch, s_2_nm_patch), axis=0)

                non_masked_indices_1 = np.concatenate((p_1_nonmask_indices_pt_coords, s_1_nonmask_indices_pt_coords), axis=0)
                non_masked_indices_2 = np.concatenate((p_2_nonmask_indices_pt_coords, s_2_nonmask_indices_pt_coords), axis=0)

                batch.append((
                    non_masked_patches_1, non_masked_indices_1, p_1_m_patch, p_1_mask_indices_pt_coords,
                    non_masked_patches_2, non_masked_indices_2, p_2_m_patch, p_2_mask_indices_pt_coords,
                    comparison_vector
                ))
            yield tuple(np.stack(t) for t in zip(*batch))

class DataModule(L.LightningDataModule):
    def __init__(self, mini_batch_size, log_batch_size, n_workers, patch_size, support_prism_prob, compare_different_series_prob):
        super().__init__()
        self.save_hyperparameters()
        self.true_batch_size = log_batch_size
        self.mini_batch_size = mini_batch_size
        self.n_workers = n_workers
        self.patch_size = patch_size
        self.support_prism_prob = support_prism_prob
        self.compare_different_series_prob = compare_different_series_prob

    def train_dataloader(self):
        data = TrainDataset(
            num_batches=100,
            mini_batch_size=self.mini_batch_size,
            n_workers=self.n_workers,
            patch_size=self.patch_size,
            support_prism_prob=self.support_prism_prob,
            compare_different_series_prob=self.compare_different_series_prob
        )

        dataloader = DataLoader(data, batch_size=None, num_workers=self.n_workers)
        return dataloader
