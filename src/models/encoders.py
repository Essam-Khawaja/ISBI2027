import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, embedding_dim))
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [batch, tabular_features]
        # output: [batch, embedding_dim]
        return self.network(x)


class Conv3DEncoder(nn.Module):
    def __init__(self, in_channels, channels, embedding_dim, dropout):
        super().__init__()

        blocks = []
        previous_channels = in_channels

        for output_channels in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=previous_channels,
                        out_channels=output_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm3d(output_channels),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2),
                )
            )
            previous_channels = output_channels

        self.blocks = nn.ModuleList(blocks)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(previous_channels, embedding_dim),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        for block in self.blocks:
            x = block(x)
            # after block k: [batch, block_channels[k], D/2, H/2, W/2]

        x = self.global_pool(x)
        # after global pool: [batch, last_channels, 1, 1, 1]

        return self.projection(x)
        # output: [batch, embedding_dim]

