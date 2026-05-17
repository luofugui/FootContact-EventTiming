import torch
import math
from torch import nn
from torch.nn.parameter import Parameter


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


class ScaledSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Sinusoidal positional encoding requires an even hidden dim.")
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32) / half_dim
        inv_freq = theta ** (-freq_seq)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        emb = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class ClassicPositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(max_len, dim)
        positions = torch.arange(0, max_len, dtype=torch.float32).view(-1, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[:, : x.shape[1]])


class FrameAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, temporal_window=2):
        super().__init__()
        self.num_heads = num_heads
        self.intra_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.inter_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_window = temporal_window

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_intra = self.intra_frame_attn(x, x, x)[0]
        x = self.norm1(x + x_intra)

        temporal_mask = self.create_temporal_mask(seq_len, self.temporal_window).to(x.device)
        expanded_mask = temporal_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
        x_inter = self.inter_frame_attn(x, x, x, attn_mask=expanded_mask)[0]
        return self.norm2(x + x_inter)

    @staticmethod
    def create_temporal_mask(seq_len, window):
        mask = torch.full((seq_len, seq_len), float("-inf"))
        for idx in range(seq_len):
            start = max(0, idx - window)
            end = min(seq_len, idx + window + 1)
            mask[idx, start:end] = 0.0
        return mask


class FootFormerSTT(nn.Module):
    """Spatial-temporal transformer block copied from the FootFormer design."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        seq_len,
        dropout=0.2,
        pos="learnable",
        mlp_dim=1024,
        temporal_window=2,
    ):
        super().__init__()
        self.pos = pos
        if pos == "learnable":
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, hidden_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == "sinusoidal":
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(hidden_dim)
        elif pos in ("classic", "standard"):
            self.positional_encodings = ClassicPositionalEncoding(hidden_dim, dropout, seq_len)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos}")

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attention": FrameAttentionBlock(
                            hidden_dim,
                            num_heads,
                            temporal_window=temporal_window,
                        ),
                        "feed_forward": nn.Sequential(
                            nn.Linear(hidden_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.pos == "learnable":
            x = x + self.positional_encodings.unsqueeze(0)
        elif self.pos == "sinusoidal":
            x = x + self.positional_encodings(x).unsqueeze(0)
        else:
            x = self.positional_encodings(x)

        for layer in self.layers:
            attended = layer["cross_attention"](x)
            x = attended + layer["feed_forward"](attended)
        return self.norm(x)


class FootFormerTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        seq_len,
        dropout=0.2,
        pos="learnable",
        mlp_dim=1024,
    ):
        super().__init__()
        self.pos = pos
        if pos == "learnable":
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, hidden_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == "sinusoidal":
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(hidden_dim)
        else:
            self.positional_encodings = ClassicPositionalEncoding(hidden_dim, dropout, seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=mlp_dim,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        if self.pos == "learnable":
            x = x + self.positional_encodings.unsqueeze(0)
        elif self.pos == "sinusoidal":
            x = x + self.positional_encodings(x).unsqueeze(0)
        else:
            x = self.positional_encodings(x)
        return self.transformer_encoder(x)


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
        transformer="multi",
        pos="learnable",
        mlp_dim=1024,
        temporal_window=2,
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
        if transformer == "multi":
            self.temporal_encoder = FootFormerSTT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=self.window_frames,
                dropout=dropout,
                pos=pos,
                mlp_dim=mlp_dim,
                temporal_window=temporal_window,
            )
        elif transformer == "transformer":
            self.temporal_encoder = FootFormerTransformer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=self.window_frames,
                dropout=dropout,
                pos=pos,
                mlp_dim=mlp_dim,
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer}")
        self.norm = nn.LayerNorm(hidden_dim)
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
        x = self.temporal_encoder(x)
        x = self.norm(x)
        return self.event_head(x)
