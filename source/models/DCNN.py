import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNStream(nn.Module):
    """
    A simple 1D CNN feature extractor for a single modality stream.
    conv_channels: list of output channels for each conv layer
    kernel_sizes: list of kernel sizes for each conv layer
    pool_sizes: list of max-pooling sizes following each conv
    dropout: dropout probability after each pool layer
    """
    def __init__(
        self,
        conv_channels,
        kernel_sizes,
        pool_sizes,
        dropout=0.5
    ):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes), \
            "conv_channels, kernel_sizes, and pool_sizes must have same length"

        layers = []
        in_ch = 1  # single-channel input per modality
        for out_ch, k, p in zip(conv_channels, kernel_sizes, pool_sizes):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=p))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)   # -> (batch, 1, seq_len)
        out = self.network(x)  # -> (batch, C_last, L_out)
        out = out.flatten(1)   # -> (batch, C_last * L_out)
        return out

class SimpleMultiStreamCNN(nn.Module):
    def __init__(self,
                 num_modalities: int,
                 conv_channels: list,
                 kernel_sizes: list,
                 pool_sizes: list,
                 mlp_hidden: int,
                 num_classes: int,
                 dropout=0.5):
        super().__init__()
        self.num_modalities = num_modalities
        self.streams = nn.ModuleList([
            CNNStream(conv_channels, kernel_sizes, pool_sizes, dropout)
            for _ in range(num_modalities)
        ])

        # register fc1 up front, but infer in_features on first call
        self.fc1 = nn.LazyLinear(mlp_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_hidden, num_classes)

    def forward(self, *xs):
        # build and concat features as before...
        feats = [self.streams[i](x) for i, x in enumerate(xs)]
        x_cat = torch.cat(feats, dim=1)

        x = self.fc1(x_cat)        # LazyLinear now has an `in_features`
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out