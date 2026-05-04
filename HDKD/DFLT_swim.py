"""
DFLT_swin.py

  - No CLS token. No DISTILL token.
  - Classification and distillation both use the same globally average-pooled
    representation [B, dim] produced after the Swin transformer.
  - Swin uses shifted-window self-attention (local → global hierarchy).
  - Window size default = 7 to match the 7×7 token grid produced by a 14×14
    feature map with patch_size=(2,2).

Output modes:
  - use_distillation=False  → pooled [B, dim]
  - use_distillation=True   → (pooled [B, dim], pooled [B, dim])
    Both outputs are the same GAP vector. StudentModel feeds them into
    head_cls and head_distill respectively, so the teacher's KL signal
    and the CE signal shape the shared representation via two gradient paths.

These return signatures match StudentModel.forward() exactly — no changes needed there.
"""

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# here 49 tokens are attending to each other
def window_partition(x, window_size):
    """
    Partition token grid into non-overlapping windows.
    Args:
        x: [B, H, W, C]
        window_size (int): window height = window width
    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # [B, nH, ws, nW, ws, C] -> [B*nH*nW, ws, ws, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

# to reconstruct back input
def window_reverse(windows, window_size, H, W):
    """
    Reverse window_partition.
    Args:
        windows: [num_windows*B, window_size, window_size, C]
        window_size (int)
        H, W: token grid dimensions
    Returns:
        x: [B, H, W, C]
    """
    B_nW = windows.shape[0]
    B = int(B_nW / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# windowed attention of complexity n2
class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with optional relative position bias.
    Supports both regular and shifted windows.

    Args:
        dim (int): token embedding dimension
        window_size (int): side length of each square attention window
        num_heads (int): number of attention heads
        dim_head (int): dimension per head
        dropout (float): attention dropout
    """
    def __init__(self, dim, window_size, num_heads, dim_head=32, dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # we use relative positional encodings here
        # Relative position bias table
        # Range: [-(ws-1), ws-1] in each dimension → (2*ws-1)^2 unique pairs
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position index for each token pair in a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        # [2, ws, ws]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        # [2, ws*ws]
        coords_flatten = coords.flatten(1)
        # [2, ws*ws, ws*ws] pairwise differences
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)        # [ws*ws, ws*ws]
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self, x):
        """
        Args:
            x: [num_windows*B, ws*ws, dim]
        Returns:
            out: [num_windows*B, ws*ws, dim]
        """
        x = self.norm(x)
        Nw, L, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Add relative position bias
        bias_index = self.relative_position_index.view(-1)
        bias = self.relative_position_bias_table[bias_index]
        bias = bias.view(L, L, self.num_heads).permute(2, 0, 1)  # [heads, L, L]
        dots = dots + bias.unsqueeze(0)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# standard feedforward network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SwinBlock(nn.Module):
    """
    Single Swin Transformer block.
    Alternates between regular window attention (shift=0) and
    shifted window attention (shift=window_size//2).

    Args:
        dim (int): embedding dimension
        H, W (int): token grid height and width
        window_size (int): attention window size
        shift (bool): whether this block uses shifted windows
        num_heads (int): number of attention heads
        dim_head (int): dimension per head
        mlp_dim (int): MLP hidden dimension
        dropout (float)
    """
    def __init__(self, dim, H, W, window_size, shift, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.H = H
        self.W = W
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.attn = WindowAttention(dim, window_size, num_heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

        # Attention mask for shifted windows — prevents cross-window attention
        if self.shift_size > 0:
            self.register_buffer('attn_mask', self._make_mask(H, W))
        else:
            self.attn_mask = None

    # this is due to circular shift and no interaction among non-adjacent pixels
    def _make_mask(self, H, W):
        """Build additive attention mask that blocks cross-region attention."""
        img_mask = torch.zeros(1, H, W, 1)
        slices_h = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        slices_w = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in slices_h:
            for w in slices_w:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # [nW, ws*ws]
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # [nW, ws*ws, ws*ws]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        """
        Args:
            x: [B, H*W, dim]  — flat patch tokens only (no extra tokens)
        Returns:
            x: [B, H*W, dim]
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W

        # Reshape to 2D grid for windowing
        x_2d = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            x_shifted = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x_2d

        # Partition into windows → [nW*B, ws, ws, C] → [nW*B, ws*ws, C]
        x_windows = window_partition(x_shifted, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        if self.attn_mask is not None:
            attn_out = self.attn(x_windows) + x_windows
        else:
            attn_out = self.attn(x_windows) + x_windows

        # Apply mask (additive, so add before softmax — mask is already embedded in attn)
        attn_out = self._masked_attn(x_windows)

        # Reverse windows → [B, H, W, C]
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x_shifted = window_reverse(attn_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_2d = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_2d = x_shifted

        x = x + x_2d.view(B, L, C)
        x = x + self.ff(x)
        return x

    def _masked_attn(self, x_windows):
        """Run window attention with optional shift mask."""
        Nw, L, C = x_windows.shape

        normed = self.attn.norm(x_windows)
        qkv = self.attn.to_qkv(normed).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attn.num_heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.attn.scale

        bias_index = self.attn.relative_position_index.view(-1)
        bias = self.attn.relative_position_bias_table[bias_index]
        bias = bias.view(L, L, self.attn.num_heads).permute(2, 0, 1)
        dots = dots + bias.unsqueeze(0)

        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            B_real = Nw // nW
            dots = dots.view(B_real, nW, self.attn.num_heads, L, L)
            dots = dots + self.attn_mask.unsqueeze(0).unsqueeze(2)
            dots = dots.view(-1, self.attn.num_heads, L, L)

        attn = self.attn.attend(dots)
        attn = self.attn.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.attn.to_out(out)


class SwinTransformer(nn.Module):
    """
    Stack of SwinBlocks alternating regular and shifted windows.

    Args:
        dim (int): embedding dimension
        depth (int): number of blocks
        H, W (int): token grid size
        window_size (int): window side length
        num_heads (int)
        dim_head (int)
        mlp_dim (int)
        dropout (float)
    """
    def __init__(self, dim, depth, H, W, window_size, num_heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            SwinBlock(
                dim=dim, H=H, W=W,
                window_size=window_size,
                shift=(i % 2 == 1),          # even=regular, odd=shifted
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: [B, L, dim]  where L = H*W (patch tokens only, no extra tokens)
        Returns:
            x: [B, L, dim]
        """
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)


class DFLT_Swin(nn.Module):
    """
    Swin-based replacement for the ViT DFLT module.

    After the Swin transformer, all patch tokens are globally average-pooled
    into a single [B, dim] vector. This vector is returned once (no distillation)
    or twice (with distillation) so StudentModel can feed it into head_cls and
    head_distill independently. Both heads receive identical input; the teacher's
    KL signal and the CE signal shape the same representation via separate gradient paths.

    Args:
        image_size (tuple): (H, W) of the feature map entering DFLT. e.g. (14, 14)
        patch_size (tuple): patch size for tokenization. e.g. (2, 2)
        dim (int): embedding dimension. Default 256
        depth (int): number of Swin blocks. Default 3
        heads (int): number of attention heads
        expansion (int): MLP expansion ratio
        channels (int): input feature map channels (e.g. 192 after FusionProjector)
        use_distillation (bool): return (pooled, pooled) if True, else pooled
        dim_head (int): dimension per head
        window_size (int): Swin window size. Default 7
        dropout (float)
        emb_dropout (float)
    """
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        expansion,
        channels,
        use_distillation=True,
        dim_head=32,
        window_size=7,
        dropout=0.,
        emb_dropout=0.,
        **kwargs,   # absorb any extra args (e.g. multi_distill) without crashing
    ):
        super().__init__()
        self.use_distillation = use_distillation

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Feature map dimensions must be divisible by patch size.'

        self.H = image_height // patch_height   # token grid height  e.g. 7
        self.W = image_width  // patch_width    # token grid width   e.g. 7
        num_patches = self.H * self.W           # e.g. 49
        patch_dim   = channels * patch_height * patch_width   # e.g. 192*2*2 = 768
        mlp_dim     = dim * expansion

        assert self.H % window_size == 0 and self.W % window_size == 0, \
            f'Token grid ({self.H}×{self.W}) must be divisible by window_size ({window_size}).'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_drop = nn.Dropout(emb_dropout)

        self.transformer = SwinTransformer(
            dim=dim,
            depth=depth,
            H=self.H,
            W=self.W,
            window_size=window_size,
            num_heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        """
        Args:
            img: [B, C, H_feat, W_feat]  e.g. [B, 192, 14, 14]

        Returns:
            use_distillation=False: pooled [B, dim]
            use_distillation=True:  (pooled [B, dim], pooled [B, dim])
                StudentModel passes these into head_cls and head_distill.
        """
        # 1. Tokenise
        x = self.to_patch_embedding(img)    # [B, N, dim]
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        # 2. Swin transformer
        x = self.transformer(x)             # [B, N, dim]

        # 3. Global average pool over all patch tokens
        pooled = self.norm(x).mean(dim=1)   # [B, dim]

        if self.use_distillation:
            return pooled, pooled           # same vector, two heads in StudentModel
        return pooled