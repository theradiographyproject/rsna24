import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from x_transformers import Encoder, TransformerWrapper, CrossAttender
from torch.nn import BCELoss
import lightning as L
from torch.optim import lr_scheduler

def divisible_by(numer, denom):
    return (numer % denom) == 0

def exists(val):
    return val is not None

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

def triple(t):
    return t if isinstance(t, tuple) else (t, t, t)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class PosClassifierHead(Module):
    def __init__(self, dim, dropout):
        super().__init__()

        self.mlp_head = nn.Linear(dim, dim)
        self.position_embedding = CrossAttender(dim = dim, depth = 1)
        self.token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(p=dropout)

        self.head = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        embedding_1 = self.position_embedding(self.token, context=x1).squeeze(1)
        embedding_2 = self.position_embedding(self.token, context=x2).squeeze(1)

        dropout_mask = self.dropout(torch.ones_like(embedding_1))

        embedding_1 = embedding_1 * dropout_mask
        embedding_2 = embedding_2 * dropout_mask

        x = torch.cat((embedding_1, embedding_2), dim=1)

        return self.head(x), embedding_1, embedding_2

class MIMHead(Module):
    def __init__(self, dim, patch_size, depth, layer_dropout):
        super().__init__()
        self.mim = CrossAttender(dim = dim, depth = depth, layer_dropout=layer_dropout)
        self.to_pixels = nn.Linear(dim, np.prod(patch_size))
        self.token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.token2 = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, pos, context1, context2):

        context1 = context1 + self.token1
        context2 = context2 + self.token2

        context = torch.cat((context1, context2), dim=1)

        unmasked = self.mim(pos, context=context)
        unmasked = self.to_pixels(unmasked)
        return unmasked

class PrismMaskEmbedder(Module):
    def __init__(
        self,
        dim,
        patch_size,
        device
    ):
        super().__init__()
        self.dim = dim
        self.device = device

        channels = 1
        self.patch_size = triple(patch_size)
        patch_dim = channels * np.prod(self.patch_size)

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim)
        )

    def create_sin_cos_embeddings_from_positions(self, positions, dim, n=1000):
        """
        Create sinusoidal and cosine embeddings from given positions.

        Parameters:
        - positions: A tensor of shape (batch_size, num_positions, 3) where the last dimension
          represents the (d, r, c) coordinates.
        - dim: The dimensionality of the embeddings to create.
        - n: A scaling factor (typically a large value like 1000).

        Returns:
        - A tensor of shape (batch_size, num_positions, dim) with the sin-cos embeddings.
        """

        arange = partial(torch.arange, device=positions.device)

        # Compute the denominators for the sin-cos embeddings
        denominators = torch.pow(n, 2 * arange(0, dim // 6) / dim).unsqueeze(0)/2  # Shape (1, dim//6)

        # Separate d, r, c components from the positions tensor
        d = positions[:, :, 0:1].float()  # Shape (batch_size, num_positions, 1)
        r = positions[:, :, 1:2].float()  # Shape (batch_size, num_positions, 1)
        c = positions[:, :, 2:3].float()  # Shape (batch_size, num_positions, 1)

        # Scale positions using denominators
        d_scaled = d * denominators
        r_scaled = r * denominators
        c_scaled = c * denominators

        # Compute sin and cos embeddings
        d_s = torch.sin(d_scaled)
        d_c = torch.cos(d_scaled)
        r_s = torch.sin(r_scaled)
        r_c = torch.cos(r_scaled)
        c_s = torch.sin(c_scaled)
        c_c = torch.cos(c_scaled)

        # Concatenate all embeddings along the last dimension
        embeddings = torch.cat((d_s, d_c, r_s, r_c, c_s, c_c), dim=-1)  # Shape (batch_size, num_positions, dim)

        return embeddings


    def forward(
        self,
        nm_patches, 
        nm_indices,
        m_indices
    ):

        nm_pos = self.create_sin_cos_embeddings_from_positions(nm_indices, self.dim)
        m_pos = self.create_sin_cos_embeddings_from_positions(m_indices, self.dim)
        nm_patch_emb = self.patch_to_embedding(nm_patches)

        return nm_pos + nm_patch_emb, m_pos

class EndToEnd(L.LightningModule):
    def __init__(self, dim, patch_size, depth, heads, pos_head_dropout, mim_depth, mim_head_layer_dropout, learning_rate, scheduler_type='linear'):
        super().__init__()
        self.save_hyperparameters()
        self.prism_mask_embedder = PrismMaskEmbedder(dim, patch_size, self.device)
        self.prism_encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            ff_glu = True    # set to true to use for all feedforwards
        )
        self.self_MIM_head = MIMHead(dim, patch_size, mim_depth, mim_head_layer_dropout)
        self.cross_MIM_head = None
        self.pos_classifier_head = PosClassifierHead(dim, pos_head_dropout)
        self.loss = BCELoss()
        self.recon_loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.T_max = 1

    def forward(self, nm_patches_1, nm_indices_1, m_indices_1, nm_patches_2, nm_indices_2, m_indices_2):

        # Take in two batches of prisms
        # Pass each batch of prisms into the PrismMaskEmbedder which returns embeddings and masked positions + patches
        embedding1, mask_pos1 = self.prism_mask_embedder(nm_patches_1, nm_indices_1, m_indices_1)
        embedding2, mask_pos2 = self.prism_mask_embedder(nm_patches_2, nm_indices_2, m_indices_2)

        # Pass the embeddings to the PrismEncoder
        encoded1 = self.prism_encoder(embedding1)
        encoded2 = self.prism_encoder(embedding2)

        # Pass the encodings to the MIMHead
        self_MIM = self.self_MIM_head(mask_pos1, encoded1, encoded2)
        self_MIM2 = self.self_MIM_head(mask_pos2, encoded2, encoded1)

        # IF SAME SCAN
        # Pass the encodings, masked positions, and final patches to the MIMDecoder and then backprop
        # Pass the encodings to the PosClassifierHead and then backprop
        # IF DIFFERENT SCAN
        # Pass the both sets of encodings, masked positions, and final patches to the CrossMIMDecoder and then backprop

        pos, penc1, penc2 = self.pos_classifier_head(encoded1, encoded2)

        return pos, self_MIM, self_MIM2

        # IF SAME SCAN
        # Pass the encodings, masked positions, and final patches to the MIMDecoder and then backprop
        # Pass the encodings to the PosClassifierHead and then backprop
        # IF DIFFERENT SCAN
        # Pass the both sets of encodings, masked positions, and final patches to the CrossMIMDecoder and then backprop

    def training_step(self, batch, batch_idx):

        non_masked_patches_1, non_masked_indices_1, p_1_m_patch, p_1_mask_indices_pt_coords, non_masked_patches_2, non_masked_indices_2, p_2_m_patch, p_2_mask_indices_pt_coords, label = batch

        label = torch.where(label > 0, torch.tensor(1.0), torch.tensor(0.0))
        label = label.type(torch.FloatTensor)

        non_masked_patches_1  = non_masked_patches_1.to('cuda:0')
        non_masked_patches_1 = non_masked_patches_1.view(non_masked_patches_1.shape[0], non_masked_patches_1.shape[1], -1)
        non_masked_indices_1 = non_masked_indices_1.to('cuda:0')
        p_1_m_patch = p_1_m_patch.to('cuda:0')
        p_1_m_patch = p_1_m_patch.view(p_1_m_patch.shape[0], p_1_m_patch.shape[1], -1)
        p_1_mask_indices_pt_coords = p_1_mask_indices_pt_coords.to('cuda:0')

        non_masked_patches_2 = non_masked_patches_2.to('cuda:0')
        non_masked_patches_2 = non_masked_patches_2.view(non_masked_patches_2.shape[0], non_masked_patches_2.shape[1], -1)
        non_masked_indices_2 = non_masked_indices_2.to('cuda:0')
        p_2_m_patch = p_2_m_patch.to('cuda:0')
        p_2_m_patch = p_2_m_patch.view(p_2_m_patch.shape[0], p_2_m_patch.shape[1], -1)
        p_2_mask_indices_pt_coords = p_2_mask_indices_pt_coords.to('cuda:0')

        label = label.to('cuda:0')

        fo, autoenc_v1, autoenc_v2 = self(
            non_masked_patches_1, non_masked_indices_1, p_1_mask_indices_pt_coords,
            non_masked_patches_2, non_masked_indices_2, p_2_mask_indices_pt_coords
        )

        #r_loss = self.recon_loss(autoenc_v1, og_inputs_1[:, 0:1, :, :])
        if (fo != fo).any():
            print("Nan detected")
            print(fo.shape)
            print(inputs_1.shape)
            print(inputs_2.shape)
            print(label.shape)

            nan_indices = torch.any(torch.isnan(fo), dim=1)

            fo = fo[~nan_indices]
            label = label[~nan_indices]
            print(fo.shape)

        fo_loss = self.loss(fo, label)
        self.log("loss", fo_loss, prog_bar=True)
        mim_loss = self.recon_loss(autoenc_v1, p_1_m_patch)
        self.log("mim_loss", mim_loss, prog_bar=True)
        mim_loss2 = self.recon_loss(autoenc_v2, p_2_m_patch)
        self.log("mim_loss2", mim_loss2, prog_bar=True)
        total_loss = mim_loss + mim_loss2 + fo_loss

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.scheduler_type == 'cosine':
          scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=0)  # eta_min is the minimum learning rate
          return {
              'optimizer': optimizer,
              'lr_scheduler': {
                  'scheduler': scheduler,
                  'interval': 'epoch',  # 'epoch' or 'step'
                  'frequency': 1,
              }
          }
        elif self.scheduler_type == 'constant':
          return optimizer
