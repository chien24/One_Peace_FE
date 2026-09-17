# Backbone video của ONE-PEACE (fine-tune trên Kinetics-400), tách ra từ
#   https://github.com/OFA-Sys/ONE-PEACE/blob/main/one_peace_vision/video/mmaction_custom/models/backbones/onepeace.py
# Thay đổi so với bản gốc (Apache 2.0, OFA-Sys):
#   - bỏ phụ thuộc mmcv / mmaction / timm (chỉ dùng để train và khởi tạo trọng số);
#   - bỏ các hàm nội suy position embedding (checkpoint K400 đã đúng 256x256, bucket 16);
#   - thêm tuỳ chọn dùng F.scaled_dot_product_attention (cùng phép tính, nhanh hơn trên GPU);
#   - thêm hàm nạp checkpoint `onepeace_video_k400.pth` và hàm lấy đặc trưng clip.
# Tên module/tham số được giữ nguyên để state_dict khớp 100% với checkpoint.

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# Chuẩn hoá ảnh dùng khi fine-tune K400 (configs/recognition/onepeace_k400.py), thứ tự RGB
IMG_MEAN = (122.771, 116.746, 104.094)
IMG_STD = (68.5, 66.632, 70.323)


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


def make_image_bucket_position(bucket_size, num_relative_distance):
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, h, w
    coords_flatten = torch.flatten(coords, 1)  # 2, h*w
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(
        size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # h*w, h*w
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index  # h*w+1, h*w+1


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


class LayerNorm2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GeGLU(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.wi_0 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.wi_1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x_gelu = self.act(self.wi_0(x))
        x_linear = self.wi_1(x)
        x = x_gelu * x_linear
        return x


class ImageAdaptor(nn.Module):
    def __init__(
        self,
        attention_heads: int = 24,
        bucket_size: int = 16,
        num_frames: int = 16,
        dropout: float = 0.0,
        embed_dim: int = 1536,
        shared_rp_bias: bool = True,
    ):
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embed_images = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, kernel_size=4, stride=4),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2D(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
        )

        scale = embed_dim ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        self.bucket_size = bucket_size
        self.num_frames = num_frames
        self.pos_embed = nn.Parameter(scale * torch.randn(bucket_size ** 2 + 1, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.shared_rp_bias = shared_rp_bias
        if shared_rp_bias:
            num_rel_dis = (2 * bucket_size - 1) * (2 * bucket_size - 1) + 3
            rp_bucket = make_image_bucket_position(bucket_size, num_rel_dis)
            self.rel_pos_table = Embedding(num_rel_dis, attention_heads, zero_init=True)
            self.register_buffer("rp_bucket", rp_bucket)

    def get_rel_pos_bias(self):
        values = F.embedding(self.rp_bucket, self.rel_pos_table.weight)
        values = values.permute(2, 0, 1).contiguous()
        return values

    def forward(self, src_images):
        batch_size = src_images.size(0)
        assert src_images.size(2) == self.bucket_size * 16, \
            f"backbone cần ảnh {self.bucket_size * 16}x{self.bucket_size * 16}"

        x = self.embed_images(src_images).flatten(2).transpose(1, 2)  # BxLxC
        cls_embedding = self.cls_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_embedding, x], dim=1)

        x = x + self.pos_embed.unsqueeze(0)
        x = self.dropout_module(x)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        self_attn_bias = self.get_rel_pos_bias() if self.shared_rp_bias else None

        return x, self_attn_bias


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.use_sdpa = use_sdpa

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.ln = nn.LayerNorm(embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query, attn_bias: Optional[Tensor] = None) -> Tensor:
        """input shape: LxBxC"""
        tgt_len, bsz, _ = query.size()

        q = self.q_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if self.use_sdpa and not self.training:
            # softmax(q k^T / sqrt(d) + bias) v — giống hệt nhánh bmm bên dưới
            if attn_bias is not None:
                attn_bias = attn_bias.to(q.dtype)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            if attn_bias is not None:
                attn_weights += attn_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout_module(attn_weights)
            attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.ln(attn)
        attn = self.out_proj(attn)
        return attn


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attention_heads: int = 24,
        bucket_size: int = 16,
        dropout: float = 0.0,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layer_scale_init_value: float = 1e-2,
        num_tadapter: int = 1,
        num_frames: int = 16,
        scale: float = 0.5,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.heads = attention_heads
        self.bucket_size = bucket_size
        self.self_attn = MultiheadAttention(embed_dim, attention_heads, use_sdpa=use_sdpa)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(0.0)

        self.image_ffn = nn.Sequential(
            GeGLU(self.embed_dim, self.ffn_embed_dim),
            self.activation_dropout_module,
            nn.LayerNorm(self.ffn_embed_dim),
            nn.Linear(self.ffn_embed_dim, self.embed_dim)
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((self.embed_dim)))

        self.MLP_Adapter = Adapter(embed_dim, skip_connect=False)
        self.S_Adapter = Adapter(embed_dim)
        self.scale = scale
        self.T_Adapter = Adapter(embed_dim, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(embed_dim)
        self.num_tadapter = num_tadapter
        self.num_frames = num_frames

    def forward(self, x, attn_bias: Optional[Tensor] = None):
        n, bt, d = x.shape
        residual = x
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        # temporal adaptation
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.self_attn(self.T_Adapter_in(self.self_attn_layer_norm(xt))))
        else:
            xt = self.T_Adapter(self.self_attn(self.self_attn_layer_norm(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + xt
        # spatial adaptation
        x = self.S_Adapter(self.self_attn(self.self_attn_layer_norm(x), attn_bias))
        x = residual + self.gamma_1 * x
        # joint adaptation
        residual = x
        xn = self.final_layer_norm(x)
        x = residual + self.gamma_2 * self.dropout_module(self.image_ffn(xn)) + \
            self.scale * self.MLP_Adapter(xn)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers: int = 40, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**layer_kwargs) for _ in range(layers)])
        self.num_layers = len(self.layers)
        self.image_layer_norm = nn.LayerNorm(layer_kwargs.get("embed_dim", 1536))

    def forward(self, image_info):
        x, attn_bias = image_info
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(0).expand(x.size(0), -1, -1, -1).flatten(0, 1)
        # (BT)xLxC -> Lx(BT)xC
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.image_layer_norm(x)
        return x


class OnePeaceViT(nn.Module):
    """Cấu hình mặc định = configs/_base_/models/onepeace.py + configs/recognition/onepeace_k400.py"""

    def __init__(
        self,
        attention_heads: int = 24,
        adapter_scale: float = 0.5,
        bucket_size: int = 16,
        num_tadapter: int = 1,
        num_frames: int = 16,
        embed_dim: int = 1536,
        ffn_embed_dim: int = 6144,
        layers: int = 40,
        layer_scale_init_value: float = 1e-2,
        shared_rp_bias: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_adapter = ImageAdaptor(
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            num_frames=num_frames,
            embed_dim=embed_dim,
            shared_rp_bias=shared_rp_bias,
        )
        self.encoder = TransformerEncoder(
            layers=layers,
            attention_heads=attention_heads,
            bucket_size=bucket_size,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            layer_scale_init_value=layer_scale_init_value,
            num_tadapter=num_tadapter,
            num_frames=num_frames,
            scale=adapter_scale,
            use_sdpa=use_sdpa,
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: B x C x T x H x W (đã chuẩn hoá)  ->  CLS token từng frame: B x D x T"""
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.encoder(self.image_adapter(x))  # L x (BT) x D
        x = x[0]                                 # CLS token: (BT) x D
        return rearrange(x, '(b t) d -> b d t', b=B, t=T)

    @torch.no_grad()
    def extract_clip_features(self, x: Tensor) -> Tensor:
        """Đặc trưng 1536-d của mỗi clip = trung bình CLS token theo thời gian.
        Đây đúng là đầu vào của I3DHead (spatial_type='avg') khi fine-tune K400."""
        return self.forward(x).mean(dim=-1)


def load_k400_checkpoint(checkpoint_path: str, use_sdpa: bool = True, with_head: bool = False):
    """Nạp onepeace_video_k400.pth (dict {'state_dict': {'backbone.*', 'cls_head.*'}}).

    Trả về model (fp32, CPU, eval) và — nếu with_head=True — lớp Linear 1536->400 của K400
    (chỉ dùng để kiểm tra nhanh tiền xử lý có đúng không).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    num_frames = backbone_sd["image_adapter.temporal_embedding"].shape[1]
    try:
        # dựng model trên 'meta' (không cấp phát RAM) rồi gán thẳng tensor của checkpoint
        with torch.device("meta"):
            model = OnePeaceViT(num_frames=num_frames, use_sdpa=use_sdpa)
        model.load_state_dict(backbone_sd, strict=True, assign=True)
    except (TypeError, AttributeError):  # torch < 2.1
        model = OnePeaceViT(num_frames=num_frames, use_sdpa=use_sdpa)
        model.load_state_dict(backbone_sd, strict=True)
    model.eval()

    if not with_head:
        return model
    head = nn.Linear(1536, state_dict["cls_head.fc_cls.weight"].shape[0])
    head.weight.data.copy_(state_dict["cls_head.fc_cls.weight"])
    head.bias.data.copy_(state_dict["cls_head.fc_cls.bias"])
    return model, head.eval()
