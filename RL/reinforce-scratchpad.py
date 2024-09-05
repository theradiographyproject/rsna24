import marimo

__generated_with = "0.8.7"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import zarr
    return mo, zarr


@app.cell
def __():
    # Given a study and a level, return the dicom (possibly subsetted and with the appropriate modification of the level's pixel value)
    return


@app.cell
def __():
    import pandas as pd

    pd.read_pickle("/cbica/home/gangarav/rsna24/dataframe.pkl")
    return pd,


@app.cell
def __(mo, pd):
    level = mo.ui.table(pd.read_csv("/cbica/home/gangarav/rsna24_raw/train_label_coordinates.csv"), selection='single')
    level
    return level,


@app.cell
def __(level, mo, zarr):
    selected_dicom = f"/cbica/home/gangarav/RSNA_PROCESSED/dicoms/{level.value.iloc[0].study_id}/{level.value.iloc[0].series_id}.zarr"
    selected_zarr = zarr.open(selected_dicom, mode='r')[:]

    s_d = level.value.iloc[0].instance_number
    s_r = int(level.value.iloc[0].y)
    s_c = int(level.value.iloc[0].x)

    d = mo.ui.slider(start=0, stop=selected_zarr.shape[0]-1, value=s_d, debounce=True)
    d
    return d, s_c, s_d, s_r, selected_dicom, selected_zarr


@app.cell
def __(d, level, mo, s_c, s_d, s_r, selected_zarr):
    mo.vstack([
        f"{level.value.iloc[0].condition} {level.value.iloc[0].level}",
        mo.image(src=selected_zarr[s_d, s_r-64:s_r+64, s_c-64:s_c+64]),
        mo.image(src=selected_zarr[d.value,:,:])
    ])

    return


@app.cell
def __():
    # RL algo initially will get access to dicom (possibly subset), condition, level and will attempt to predict instance, x, y (possibly subset) 

    # RL algo can request data with center point (instance, x, y) and size (d, x(pow2), y(pow2), num_patches

    # reward:
    # - negative proportional to numberof patches and size
    # - positive if center matches at the end (d +/- 1, x and y are +- 10)
    return


@app.cell
def __(SubsetRandomSampler, datasets, np, plot_images, torch, transforms):
    def get_train_valid_loader(
        data_dir,
        batch_size,
        random_seed,
        valid_size=0.1,
        shuffle=True,
        show_sample=False,
        num_workers=4,
        pin_memory=False,
    ):
        """Train and validation data loaders.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args:
            data_dir: path directory to the dataset.
            batch_size: how many samples per batch to load.
            random_seed: fix seed for reproducibility.
            valid_size: percentage split of the training set used for
                the validation set. Should be a float in the range [0, 1].
                In the paper, this number is set to 0.1.
            shuffle: whether to shuffle the train/validation indices.
            show_sample: plot 9x9 sample grid of the dataset.
            num_workers: number of subprocesses to use when loading the dataset.
            pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
                True if using GPU.
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (valid_size >= 0) and (valid_size <= 1), error_msg

        # define transforms
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        trans = transforms.Compose([transforms.ToTensor(), normalize])

        # load dataset
        dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=9,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy()
            X = np.transpose(X, [0, 2, 3, 1])
            plot_images(X, labels)

        return (train_loader, valid_loader)
    return get_train_valid_loader,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
