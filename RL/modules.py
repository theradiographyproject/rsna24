import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
import numpy as np

import sys
import os
# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to sys.path
sys.path.append(parent_dir)
from modelEmbedding import EndToEnd
from datasetTrain import patchify

class PrismExtractor:
    """Extract a sub-prism from a dicom.

    Extracts a prism `phi` around location `l`
    from an image `x` with a maximum of `n` patches.

    Args:
        x: a 4D Tensor of shape (B, D, R, C). The minibatch
            of series.
        l: a 2D Tensor of shape (B, z, y, x, d, r, c). Contains normalized
            coordinates in the range [-1, 1].

    Returns:
        phi: a 4D tensor of shape (B, d, r, c). The
            foveated glimpse of the image.
    """

    def __init__(self):
        self.model = EndToEnd.load_from_checkpoint("/cbica/home/gangarav/rsna24/checkpoint_kai/09092024_transfer/epoch=14-step=149760.ckpt")
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.prism_encoder.eval()
        self.model.self_MIM_head.eval()
        self.model.pos_classifier_head.eval()

    def emb_forward(self, nm_patches, nm_indices, m_indices):
        with torch.no_grad():
            embedding, _ = self.model.prism_mask_embedder(nm_patches, nm_indices, m_indices)
            encoded = self.model.prism_encoder(embedding)
            pos = self.model.pos_classifier_head.position_embedding(self.model.pos_classifier_head.token, context=encoded).squeeze(1)
            return pos

    def load_prism_from_arr(self, arr, prism_indices, patch_size, spacing, M, N, origin=None):
        # assume that arr has already been normalized
        slices = [slice(*s) for s in prism_indices]
        prism = arr[tuple(slices)].unsqueeze(0)
        midpoint_arr_coords = np.array([np.mean(i_pairs) for i_pairs in prism_indices]) - [0.5, 0.5, 0.5]
      
        offset = [0, 0, 0] if origin is None else midpoint_arr_coords - origin
      
        m_patch, nm_patch, sk_patch, \
        mask_indices, nonmask_indices, skipped_indices, \
        mask_indices_pt_coords, nonmask_indices_pt_coords, skipped_indices_pt_coords \
        = patchify(prism, offset, patch_size, spacing, 0, N)

        nm_patch = nm_patch.unsqueeze(0)
        nm_patch = nm_patch.view(nm_patch.shape[0], nm_patch.shape[1], -1)

        return nm_patch, torch.tensor(nonmask_indices).unsqueeze(0), torch.tensor(mask_indices).unsqueeze(0), midpoint_arr_coords
  
    def extract_prism_emb(self, X, I, L):
        """Extract sub-prism from `x` as described by `l`.

        Args:
        X: a 4D Tensor of shape (B, D, R, C). The minibatch
            of series.
        L: a 2D Tensor of shape (B, 6) (z, y, x, d, r, c). 
           z, y, x are normalized coordinates in the range [-1, 1].
           d, r, c specify the dimensions of the prism to extract.
        I: metadata tensor

        Returns:
            prism: a 4D Tensor of shape (B, d, r, c)
        """
        B, D, R, C = X.shape
        embs = []
        
        for b in range(B):
            #z, y, x, d, r, c = L[b]
            z, y, x = L[b]
            d, r, c = 4, 64, 64
            #
            slice_indices = self.get_subsample_slices(z, y, x, d, r, c, D, R, C)
            
            nm_patch, nm_indices, m_indices, midpoint_arr_coords = self.load_prism_from_arr(X[b], slice_indices, (1, 16, 16), (4, 1, 1), 0, 100)
            nm_indices = nm_indices.to('cuda' if torch.cuda.is_available() else 'cpu')
            m_indices = m_indices.to('cuda' if torch.cuda.is_available() else 'cpu')
            emb = self.emb_forward(nm_patch, nm_indices, m_indices)
            embs.append(emb)

        return torch.stack(embs)

    def get_subsample_slices(self, z, y, x, d, r, c, D, R, C):
        # Convert normalized coordinates (-1 to 1) to array indices using PyTorch functions
        z_idx = int(torch.round((z + 1) * (D - 1) / 2).item())
        y_idx = int(torch.round((y + 1) * (R - 1) / 2).item())
        x_idx = int(torch.round((x + 1) * (C - 1) / 2).item())
    
        # Compute the initial start indices
        z_start = z_idx - d // 2
        y_start = y_idx - r // 2
        x_start = x_idx - c // 2
    
        # Adjust the start indices to ensure they are within bounds and the size is preserved
        z_start = max(0, min(z_start, D - d))
        y_start = max(0, min(y_start, R - r))
        x_start = max(0, min(x_start, C - c))
    
        # Compute the end indices to maintain the desired size
        z_end = z_start + d
        y_end = y_start + r
        x_end = x_start + c
    
        return ((z_start, z_end), (y_start, y_end), (x_start, x_end))


class Embedder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        model = EndToEnd.load_from_checkpoint("/cbica/home/gangarav/rsna24/checkpoint_kai/09092024_transfer/epoch=14-step=149760.ckpt")
        model.eval()
        model.prism_encoder.eval()
        model.self_MIM_head.eval()
        model.pos_classifier_head.eval()

    def forward(self, x):
        return torch.randn(x.shape[0], self.d).to(torch.device("cuda"))
  


class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 6). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, h_g, h_l):
        super().__init__()

        self.prismExtractor = PrismExtractor()

        #glimpse layer (embedding model, will return the position embedding vector, this model should not be backpropped)
        D_in = 768 # FC from embedder
        self.fc1 = nn.Linear(D_in, h_g)

        # r = random.randint(6, 8)
        # c = random.randint(6, 8)
        # d = random.randint(1, 3)
      
        # location layer
        D_in = 3
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, spacing, l_t_prev):
        # generate glimpse phi from image x
        emb = self.prismExtractor.extract_prism_emb(x, spacing, l_t_prev)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(emb))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class CoreNetwork(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 3).
        l_t: a 2D vector of shape (B, 3).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t, std):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))
        print(mu)

        # reparametrization trick
        l_t = torch.distributions.Normal(mu, std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t