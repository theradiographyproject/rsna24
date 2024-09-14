import marimo

__generated_with = "0.8.7"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import zarr
    return mo, zarr


@app.cell
def __():
    import numpy as np
    import random
    import torch
    return np, random, torch


@app.cell
def __():
    from modules import PrismExtractor, GlimpseNetwork
    from data_loader import get_train_valid_loader
    return GlimpseNetwork, PrismExtractor, get_train_valid_loader


@app.cell
def __(get_train_valid_loader, torch):
    import os
    import time
    import shutil
    import pickle

    import torch.nn.functional as F

    from tqdm import tqdm
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    #from tensorboard_logger import configure, log_value

    from model import RecurrentAttention
    from utils import AverageMeter


    class Trainer:
        """A Recurrent Attention Model trainer.

        All hyperparameters are provided by the user in the
        config file.
        """

        def __init__(self, data_loader):
            """
            Construct a new Trainer instance.

            Args:
                config: object containing command line arguments.
                data_loader: A data iterator.
            """
            self.num_train=1000


            self.device = torch.device("cuda")

            self.loc_hidden = 32
            self.glimpse_hidden = 32

            # core network params
            self.num_glimpses = 2
            self.hidden_size = 64

            # reinforce params
            self.std = 0.1
            self.M = 1

            self.train_loader = data_loader[0]

            # training params
            self.epochs = 100
            self.start_epoch = 0
            self.momentum = 0.5
            self.lr = 3e-3

            self.lr_patience = 20 
            # # misc params
            # self.best = config.best
            # self.ckpt_dir = config.ckpt_dir
            # self.logs_dir = config.logs_dir
            # self.best_valid_acc = 0.0
            # self.counter = 0
            # self.lr_patience = config.lr_patience
            # self.train_patience = config.train_patience
            # self.use_tensorboard = config.use_tensorboard
            # self.resume = config.resume
            # self.print_freq = config.print_freq
            # self.plot_freq = config.plot_freq
            # self.model_name = "ram_{}_{}x{}_{}".format(
            #     config.num_glimpses,
            #     config.patch_size,
            #     config.patch_size,
            #     config.glimpse_scale,
            # )

            # self.plot_dir = "./plots/" + self.model_name + "/"
            # if not os.path.exists(self.plot_dir):
            #     os.makedirs(self.plot_dir)

            # # configure tensorboard logging
            # if self.use_tensorboard:
            #     tensorboard_dir = self.logs_dir + self.model_name
            #     print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            #     if not os.path.exists(tensorboard_dir):
            #         os.makedirs(tensorboard_dir)
            #     configure(tensorboard_dir)

            # build RAM model
            self.model = RecurrentAttention(
                self.loc_hidden,
                self.glimpse_hidden,
                self.std,
                self.hidden_size,
            )
            self.model = self.model.float().to(self.device)

            # initialize optimizer and scheduler
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", patience=self.lr_patience
            )

        def reset(self):
            h_t = torch.zeros(
                self.batch_size,
                self.hidden_size,
                dtype=torch.float,
                device=self.device,
                requires_grad=True,
            )
            l_t = torch.FloatTensor(self.batch_size, 3).uniform_(-1, 1).to(self.device)
            l_t.requires_grad = True

            return h_t, l_t

        def train(self):
            """Train the model on the training set.

            A checkpoint of the model is saved after each epoch
            and if the validation accuracy is improved upon,
            a separate ckpt is created for use on the test set.
            """
            # load the most recent checkpoint
            if self.resume:
                self.load_checkpoint(best=False)

            print(
                "\n[*] Train on {} samples, validate on {} samples".format(
                    self.num_train, self.num_valid
                )
            )

            for epoch in range(self.start_epoch, self.epochs):

                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                        epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                    )
                )

                # train for 1 epoch
                train_loss, train_acc = self.train_one_epoch(epoch)

                # evaluate on validation set
                valid_loss, valid_acc = self.validate(epoch)

                # # reduce lr if validation loss plateaus
                self.scheduler.step(-valid_acc)

                is_best = valid_acc > self.best_valid_acc
                msg1 = "train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
                if is_best:
                    self.counter = 0
                    msg2 += " [*]"
                msg = msg1 + msg2
                print(
                    msg.format(
                        train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                    )
                )

                # check for improvement
                if not is_best:
                    self.counter += 1
                if self.counter > self.train_patience:
                    print("[!] No improvement in a while, stopping training.")
                    return
                self.best_valid_acc = max(valid_acc, self.best_valid_acc)
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "best_valid_acc": self.best_valid_acc,
                    },
                    is_best,
                )

        def train_one_epoch(self, epoch):
            """
            Train the model for 1 epoch of the training set.

            An epoch corresponds to one full pass through the entire
            training set in successive mini-batches.

            This is used by train() and should not be called manually.
            """
            self.model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()
            accs = AverageMeter()

            tic = time.time()
            with tqdm(total=self.num_train) as pbar:
                for i, (x, spacing, y) in enumerate(self.train_loader):

                    self.optimizer.zero_grad()

                    x, spacing, y = x.to(self.device).float(), spacing.to(self.device).float(), y.to(self.device).float()

                    # initialize location vector and hidden state
                    self.batch_size = x.shape[0]
                    h_t, l_t = self.reset()

                    # extract the glimpses
                    locs = []
                    log_pi = []
                    baselines = []
                    for t in range(self.num_glimpses):
                        # forward pass through model
                        h_t, l_t, b_t, p = self.model(x, spacing, l_t, h_t, self.std * (0.7 ** epoch))

                        # store
                        locs.append(l_t)
                        baselines.append(b_t)
                        log_pi.append(p)

                    # Convert lists to tensors and reshape
                    baselines = torch.stack(baselines).transpose(1, 0).squeeze(2)  # [batch_size, num_glimpses]
                    log_pi = torch.stack(log_pi).transpose(1, 0)        # [batch_size, num_glimpses, location]
                    log_pi = torch.sum(log_pi, dim=2)


                    # Calculate reward based on the negative squared Euclidean distance
                    final_l_t = l_t  # l_t from the last time step
                    squared_distance = torch.sum(((final_l_t - y)*spacing) ** 2, dim=1)  # [batch_size]
                    R = -squared_distance  # [batch_size]
                    R = R.unsqueeze(1).repeat(1, self.num_glimpses)  # Match the number of glimpses
                    print(((final_l_t - y)*spacing), l_t, y)


                    # Compute losses for differentiable modules
                    # Action loss: Mean squared error between final_l_t and y
                    loss_action = F.mse_loss(final_l_t, y).float()

                    # Baseline loss: Mean squared error between baselines and reward
                    loss_baseline = F.mse_loss(baselines, R).float()

                    # REINFORCE loss: Policy gradient loss
                    adjusted_reward = R - baselines.detach()

                    loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)  # Sum over time steps
                    loss_reinforce = torch.mean(loss_reinforce, dim=0)  # Mean over batch
                    loss_reinforce = loss_reinforce.float()

                    # Total loss
                    loss = loss_action + loss_baseline + loss_reinforce * 0.01  # Adjust the scale if needed

                    # Compute a metric similar to accuracy (negative mean squared distance)
                    acc = -squared_distance.mean()

                    # store
                    losses.update(loss.item(), x.size(0))
                    accs.update(acc.item(), x.size(0))

                    # compute gradients and update SGD
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    toc = time.time()
                    batch_time.update(toc - tic)

                    pbar.set_description(
                        (
                            "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                                (toc - tic), loss.item()*1000, acc.item()
                            )
                        )
                    )
                    pbar.update(self.batch_size)

                return losses.avg, accs.avg

    trnr = Trainer(get_train_valid_loader())
    return (
        AverageMeter,
        F,
        RecurrentAttention,
        ReduceLROnPlateau,
        Trainer,
        os,
        pickle,
        shutil,
        time,
        tqdm,
        trnr,
    )


@app.cell
def __():
    import pandas as pd
    df = pd.read_csv("/cbica/home/gangarav/rsna24_raw/train_label_coordinates.csv")
    df = df[df["condition"] == "Spinal Canal Stenosis"]
    df = df[df["level"] == "L5/S1"]
    df[:10]
    return df, pd


@app.cell
def __(np, pe, torch):
    x, y, z = np.mgrid[0:100, 0:100, 0:100]
    X = x+y+z
    X = np.expand_dims(X, axis=0)
    pe.extract_prism(torch.tensor(X), torch.tensor([(-1, -1, -1, 4, 64, 64)]))
    return X, x, y, z


@app.cell
def __():
    1.01*17/2
    return


@app.cell
def __(np):
    a = np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    b, c, d, e, f, g = a[0]
    return a, b, c, d, e, f, g


@app.cell
def __(b, c, d):
    b, c, d
    return


@app.cell
def __():
    # Given a study and a level, return the dicom (possibly subsetted and with the appropriate modification of the level's pixel value)
    return


@app.cell
def __(pd):
    o = pd.read_pickle("/cbica/home/gangarav/rsna24/dataframe.pkl")
    o
    return o,


@app.cell
def __(mo, pd):
    level = mo.ui.table(pd.read_csv("/cbica/home/gangarav/rsna24_raw/train_label_coordinates.csv"), selection='single')
    level
    return level,


@app.cell
def __(level, o):
    roow = o[o['series'] == str(level.value.iloc[0].series_id)].iloc[0]
    roow['z_spacing'], roow['y_spacing'], roow['x_spacing']
    return roow,


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


app._unparsable_cell(
    r"""
    np.mean(i_pairs) for i_pairs in [()]
    """,
    name="__"
)


@app.cell
def __(trnr):
    trnr.train_one_epoch(2)
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
