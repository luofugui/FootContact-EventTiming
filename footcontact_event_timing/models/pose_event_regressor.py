import torch
from torch import nn


class PoseEventRegressor(nn.Module):
    """Predict an event offset inside a fixed pose window."""

    def __init__(
        self,
        num_joints,
        joint_dim,
        window_frames,
        num_channels=4,
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.joint_dim = int(joint_dim)
        self.window_frames = int(window_frames)
        self.input_dim = self.num_joints * self.joint_dim

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.pose_embed = nn.Linear(self.input_dim, hidden_dim)
        self.event_embed = nn.Embedding(2, hidden_dim)
        self.channel_embed = nn.Embedding(num_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.window_frames, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joints, event_type, channel):
        batch_size, frames, joints_count, joint_dim = joints.shape
        if frames != self.window_frames:
            raise ValueError(f"Expected {self.window_frames} frames, got {frames}.")
        if joints_count != self.num_joints or joint_dim != self.joint_dim:
            raise ValueError(
                f"Expected joints shape (*, {self.num_joints}, {self.joint_dim}), "
                f"got (*, {joints_count}, {joint_dim})."
            )

        x = joints.reshape(batch_size, frames, -1)
        x = self.pose_embed(self.input_norm(x))
        x = x + self.pos_embed[:, :frames]
        x = x + self.event_embed(event_type).unsqueeze(1)
        x = x + self.channel_embed(channel).unsqueeze(1)
        x = self.encoder(x)
        return torch.sigmoid(self.head(x.mean(dim=1))).squeeze(-1)
