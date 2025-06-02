import math

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn import init
from torch.nn import functional as F
from tqdm import tqdm
from visualizer import get_local




# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.W1 = nn.Linear(dim, hidden_dim)  
        self.W2 = nn.Linear(hidden_dim, dim)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W1(x)    
        x = self.relu(x)  
        x = self.W2(x)    
        return self.dropout(x)  



def get_pos_embedding(pos_idxs, embedding_dim, device, freq=10000):
    # Initialize position encoding matrix, device set to GPU
    pos_embeddings = torch.zeros(len(pos_idxs), embedding_dim).to(device)

    for i in tqdm(range(len(pos_idxs)), desc="Processing"):
        pos_idx = pos_idxs[i]

        for dim in range(0, embedding_dim, 2):
            # Calculation of sin and cos
            sinusoid = torch.sin(pos_idx / freq ** (dim / embedding_dim))
            cosinusoid = torch.cos(pos_idx / freq ** (dim / embedding_dim))

            # Fill the embedding matrix
            pos_embeddings[i, dim] = sinusoid
            pos_embeddings[i, dim + 1] = cosinusoid


    return pos_embeddings



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
 
        nn.init.xavier_uniform_(self.to_qkv.weight)

    @get_local('attention_map2')
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attention_map2 = attn

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        super(Embeddings, self).__init__()
        # Embedding layer
        self.lut = nn.Linear(vocab, dim, bias=True,  dtype=torch.float32)
        # Embedded dimension
        self.d_model = dim

    def forward(self, x):
        # Returns the embedding matrix corresponding to x (needs to be multiplied by math.sqrt(d_model))
        return self.lut(x) * math.sqrt(self.d_model)


class clsDNA(nn.Module):
    def __init__(self, *, seq_len, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', vocab = 40, dim_head = 64, dropout = 0., emb_dropout = 0., get_last_feature = False, use_auto_pos = False, snp_count_list = None):
        super().__init__()

        self.dim = dim
        self.pool = pool
        # self.pool = 'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        print("model type:", self.pool)

        self.use_auto_pos = use_auto_pos
        self.embedding_layer = Embeddings(vocab,dim)


        if snp_count_list is not None:
            pos_embedding_list = []
            for snp_count in snp_count_list:
                pos = nn.Parameter(torch.randn(1, 1, dim))
                pos = pos.expand(1, snp_count, dim)
                pos_embedding_list.append(pos)
            self.pos_embedding = nn.Parameter(torch.cat(pos_embedding_list, dim=1))

        if pool == 'cls' and snp_count_list is None and use_auto_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len+1, dim))
        elif pool == 'mean' and snp_count_list is None and use_auto_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)


        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )


        self.get_last_feature = get_last_feature

    def forward(self, snp_data, pos_data):
        snp_data = snp_data.to(torch.float32)
        pos_data = pos_data.to(torch.float32)

        x = self.embedding_layer(snp_data)

        if not pos_data.shape == x.shape:
            pos_data = pos_data.expand_as(x)
        x = pos_data + x


        b, n, _ = x.shape
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.use_auto_pos:
            x = self.pos_embedding + x
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        output = self.mlp_head(x)

        if self.get_last_feature:
            return output, x

        return output, None

class clsDNA_no_pos(nn.Module):
    def __init__(self, *, seq_len, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', vocab = 40, dim_head = 64, dropout = 0., emb_dropout = 0., get_last_feature = False, use_auto_pos = False, snp_count_list = None):
        super().__init__()

        self.dim = dim
        self.pool = pool
        # self.pool = 'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        print("model type:", self.pool)

        self.use_auto_pos = use_auto_pos
        self.embedding_layer = Embeddings(vocab,dim)


        if snp_count_list is not None:
            pos_embedding_list = []
            for snp_count in snp_count_list:
                pos = nn.Parameter(torch.randn(1, 1, dim))
                pos = pos.expand(1, snp_count, dim)
                pos_embedding_list.append(pos)
            self.pos_embedding = nn.Parameter(torch.cat(pos_embedding_list, dim=1))

        if pool == 'cls' and snp_count_list is None and use_auto_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len+1, dim))
        elif pool == 'mean' and snp_count_list is None and use_auto_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)


        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )


        self.get_last_feature = get_last_feature

    def forward(self, snp_data, pos_data):
        snp_data = snp_data.to(torch.float32)
        pos_data = pos_data.to(torch.float32)

        x = self.embedding_layer(snp_data)


        b, n, _ = x.shape
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.use_auto_pos:
            x = self.pos_embedding + x
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        output = self.mlp_head(x)

        if self.get_last_feature:
            return output, x

        return output, None

