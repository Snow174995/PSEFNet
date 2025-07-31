import argparse
import torch
import torch.nn as nn
from general import calculate_laplacian_with_self_loop, get_laplacian, normalize,GLU
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
from mts import CNN1D, GAR, VAR, CNNRNN, CNNRNNRes
from DSANet import SelfAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PSEFNet(nn.Module):
    def __init__(self, config):
        super(PSEFNet, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.seq_len = config['seq_len']

        self.pse_module = PSEModule(self.input_dim, pos_dim=config['pos_dim'], hidden_dim=self.hidden_dim)
        self.fusion_module = EmotionStructureFusion(self.hidden_dim)
        self.context_module = StructureContextFusion(self.hidden_dim)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Linear(self.hidden_dim, 1)  

    def forward(self, x_embed, structure_vec):
        """
        : param x_embed: Comment Embedding (B, L, D)
        : param structure_vec: Structural Vector (B, D)
        : return: Score Prediction (B, 1)
        """
        x = self.pse_module(x_embed)  # Sentiment Enhancement
        x = self.fusion_module(x, x_embed)  # Structural Information Fusion
        x = self.context_module(x, structure_vec)  # Contextual Structure Modeling
        x_pool = self.pooling(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        score_pred = self.predictor(x_pool)  # Score Prediction
        return score_pred

class PSEModule(nn.Module):
    def __init__(self, input_dim, pos_dim, hidden_dim):
        super(PSEModule, self).__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, 128, pos_dim))  # L=128
        self.sentiment_attention = nn.Linear(input_dim + pos_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, L, D)
        pos_embed = self.position_embedding.repeat(x.size(0), 1, 1)
        x_aug = torch.cat([x, pos_embed], dim=-1)  # (B, L, D+P)
        h = torch.tanh(self.sentiment_attention(x_aug))  # (B, L, H)
        attn = torch.softmax(self.score(h), dim=1)  # (B, L, 1)
        x_out = x * attn  # Position-Sentiment Guided Weighting
        return x_out

class EmotionStructureFusion(nn.Module):
    def __init__(self, dim):
        super(EmotionStructureFusion, self).__init__()
        self.key_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, structure_embed):
        # x: (B, L, D) Comment Feature Representation；structure_embed: (B, L, D) 
        Q = self.query_proj(x)
        K = self.key_proj(structure_embed)
        V = self.value_proj(structure_embed)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1)**0.5)
        weights = torch.softmax(scores, dim=-1)
        fusion = torch.matmul(weights, V)
        return self.out_proj(fusion + x)  # Residual Fusion


class StructureContextFusion(nn.Module):
    def __init__(self, dim):
        super(StructureContextFusion, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x, structure_vec):
        # x: (B, L, D)，structure_vec: (B, D)
        structure_broadcast = structure_vec.unsqueeze(1).expand(-1, x.size(1), -1)
        x_structure = x + structure_broadcast
        conv_input = x_structure.transpose(1, 2)  # for Conv1D
        conv_out = self.conv(conv_input).transpose(1, 2)
        trans_out = self.transformer(x)

        fusion_input = torch.cat([trans_out, conv_out], dim=-1)
        gate = torch.sigmoid(self.gate(fusion_input))
        out = gate * conv_out + (1 - gate) * trans_out
        return out
