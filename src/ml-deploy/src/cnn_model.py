"""
cnn_model.py
=============
Stage: DAY 3 (Stage 3) — the locked CNN architecture.

    3x3 surface patch -> CNN Encoder -> learned embedding -> MLP Decoder
    -> 15-depth temperature profile

Kept deliberately small (per the brief) since the input patch is only
3x3 pixels — there is no room for deep/large receptive fields here.
Every architectural knob (filters, embedding dim, hidden dim,
activation) is configurable; nothing about the channel count or depth
count is hard-coded, both are passed in by the caller (train_cnn.py),
which reads them from the Day 1 dataset at runtime.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _get_activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"[cnn_model] Unknown activation '{name}'. "
                          f"Choose from {list(_ACTIVATIONS.keys())}.")
    return _ACTIVATIONS[name]


class CNNEncoder(nn.Module):
    """
    3x3 patch (n_input_channels, 3, 3) -> learned embedding (embedding_dim,).

    Conv2D(kernel=3, padding=1) preserves the 3x3 spatial size through
    both conv layers (there's no room to downsample a 3x3 input further
    without losing the whole patch), then flattens and projects to the
    embedding with one Linear layer. The embedding itself is left
    WITHOUT a final activation (a plain linear projection) so it stays
    a natural, unclipped representation - better suited for later
    PCA/t-SNE/UMAP visualization than a ReLU-clipped one.
    """

    def __init__(self, n_input_channels: int, patch_size: int = 3,
                 num_filters: int = 16, embedding_dim: int = 32,
                 activation: str = "relu"):
        super().__init__()
        act_cls = _get_activation(activation)

        self.conv1 = nn.Conv2d(n_input_channels, num_filters, kernel_size=3, padding=1)
        self.act1 = act_cls()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.act2 = act_cls()
        self.flatten = nn.Flatten()
        self.fc_embed = nn.Linear(num_filters * patch_size * patch_size, embedding_dim)

        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_input_channels, patch_size, patch_size)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.flatten(x)
        embedding = self.fc_embed(x)
        return embedding  # (batch, embedding_dim)


class MLPDecoder(nn.Module):
    """embedding -> hidden -> hidden -> 15-depth temperature profile."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64,
                 n_output_depths: int = 15, activation: str = "relu"):
        super().__init__()
        act_cls = _get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, n_output_depths),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class PatchCNN(nn.Module):
    """
    Full Stage 3 model: CNNEncoder -> MLPDecoder.

    model.encode(x) exposes the embedding directly (e.g. for later
    PCA/t-SNE analysis) without confusing it with the final 15-depth
    output of model.forward(x).
    """

    def __init__(self, n_input_channels: int, patch_size: int = 3,
                 num_filters: int = 16, embedding_dim: int = 32,
                 hidden_dim: int = 64, n_output_depths: int = 15,
                 activation: str = "relu"):
        super().__init__()
        self.encoder = CNNEncoder(n_input_channels, patch_size, num_filters,
                                   embedding_dim, activation)
        self.decoder = MLPDecoder(embedding_dim, hidden_dim, n_output_depths, activation)

        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_output_depths = n_output_depths
        self.activation = activation

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the learned embedding only, shape (batch, embedding_dim).
        NOT the final 15-depth prediction - use forward()/predict for that."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_input_channels, patch_size, patch_size)
        embedding = self.encoder(x)
        return self.decoder(embedding)  # (batch, n_output_depths)
