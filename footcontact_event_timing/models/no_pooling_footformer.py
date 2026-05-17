import torch
from torch import nn
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """Temporal graph convolution used by the original FootFormer pose embedder."""

    def __init__(self, in_features, out_features, node_n, bias=True):
        super().__init__()
        self.weight = Parameter(torch.empty(in_features, out_features))
        self.att = Parameter(torch.empty(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class PoseEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, pose_embedder="gcn"):
        super().__init__()
        self.pose_embedder = pose_embedder
        if pose_embedder == "gcn":
            self.embedding = GraphConvolution(input_dim, hidden_dim, node_n=seq_len)
        elif pose_embedder == "linear":
            self.embedding = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown pose_embedder: {pose_embedder}")

    def forward(self, x):
        return self.embedding(x)


class NoPoolingFootFormerEventDetector(nn.Module):
    """FootFormer-style temporal event detector without temporal pooling."""

    def __init__(
        self,
        num_joints,
        joint_dim,
        window_frames,
        num_event_classes,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
        pose_embedder="gcn",
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        self.joint_dim = int(joint_dim)
        self.window_frames = int(window_frames)
        self.num_event_classes = int(num_event_classes)
        input_dim = self.num_joints * self.joint_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.pose_embed = PoseEmbedder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            seq_len=self.window_frames,
            pose_embedder=pose_embedder,
        )
        self.dropout = nn.Dropout(dropout)
        self.pre_encoder_norm = nn.LayerNorm(hidden_dim)
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
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.event_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_event_classes),
        )

    def forward(self, joints):
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
        x = self.dropout(x)
        x = self.pre_encoder_norm(x)
        x = x + self.pos_embed[:, :frames]
        x = self.temporal_encoder(x)
        return self.event_head(x)
