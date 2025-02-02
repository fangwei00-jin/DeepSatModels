import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from loguru import logger


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormLocal(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # logger.info('before fn: ', x.shape)
        x = self.fn(x, **kwargs)
        # logger.info('after fn: ', x.shape)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.): # 128, 4, 32, 0
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): #torch.Size([2304, 79, 128]) -> 2304 79 16 8
        # logger.info(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # logger.info(q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MultiScaleAttention4D(nn.Module):
    def __init__(self, patchsize=[(4, 4), (2, 2)], d_input=79, d_model=79):
        logger.info(f'patchsize: {patchsize}, d_input:{d_input}, d_model:{d_model}')
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                                           nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = rearrange(x, 'b t (h w) -> b t h w', h = 16)
        b, _, h, w = x.size()
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        # logger.info('shape of q k v_', _query.shape, _key.shape, _value.shape)
        for (width, height), query, key, value in zip(
                self.patchsize,
                torch.chunk(_query, len(self.patchsize), dim=1),
                torch.chunk(_key,   len(self.patchsize), dim=1),
                torch.chunk(_value, len(self.patchsize), dim=1)
                ):
            d_k = query.shape[-3]
            out_w, out_h = w // width, h // height

            query = query.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5)
            query = query.contiguous().view(b, out_h * out_w, d_k * height * width)
            key = key.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5)
            key = key.contiguous().view(b, out_h * out_w, d_k * height * width)
            value = value.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5)
            value = value.contiguous().view(b, out_h * out_w, d_k * height * width)

            scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** -0.5)
            p_attn = scores.softmax(dim=-1)
            y = torch.matmul(p_attn, value)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        x = rearrange(x, 'b t h w -> b t (h w)')
        return x

class MultiScaleAttention5D(nn.Module):
    def __init__(self, patchsize=[(4, 4), (2, 2)], d_input=79, d_model=79):
        logger.info(f'patchsize: {patchsize}, d_input:{d_input}, d_model:{d_model}')
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_input, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                                           nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        output = []
        _query = rearrange(self.query_embedding(x), 'b (t c) h w -> b t c h w', c=c)
        _key = rearrange(self.key_embedding(x), 'b (t c) h w -> b t c h w', c=c)
        _value = rearrange(self.value_embedding(x), 'b (t c) h w -> b t c h w', c=c)
        # logger.info('shape of q k v_', _query.shape, _key.shape, _value.shape)
        for (height, width), query, key, value in zip(
                self.patchsize,
                torch.chunk(_query, len(self.patchsize), dim=2),
                torch.chunk(_key,   len(self.patchsize), dim=2),
                torch.chunk(_value, len(self.patchsize), dim=2)
                ):
            c = query.shape[-3]
            out_w, out_h = w // width, h // height

            # query = query.view(b, t, c, out_h, height, out_w, width).permute(0, 1, 3, 5, 2, 4, 6)
            # query = query.contiguous().view(b, t * out_h * out_w, c * height * width)
            query = rearrange(query, 'b t c (oh h) (ow w) -> b (t oh ow) (c h w)', ow=out_w, oh=out_h)
            key = rearrange(key, 'b t c (oh h) (ow w) -> b (t oh ow) (c h w)', ow=out_w, oh=out_h)
            value = rearrange(value, 'b t c (oh h) (ow w) -> b (t oh ow) (c h w)', ow=out_w, oh=out_h)

            scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** -0.5)
            p_attn = scores.softmax(dim=-1)
            y = torch.matmul(p_attn, value)
            # 3) "Concat" using a view and apply a final linear.
            y = rearrange(y, 'b (t oh ow) (c h w) -> b t c (oh h) (ow w)', t=t, c=c, oh=out_h, ow=out_w, w=width, h=height)
            # y = y.view(b, out_h, out_w, c, height, width)
            # y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)
            output.append(y)
        output = torch.cat(output, 2)
        output = rearrange(output, 'b t c h w -> b (t c) h w')
        x = self.output_linear(output)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=t)
        return x
    pass

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, bias=None, p=64, g=64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        # print(__class__.__name__, in_channels, out_channels, kernel_size, stride, p, g)
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             groups=1, padding=kernel_size // 3, stride=stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=p, stride=stride)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class ReAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class LeFF(nn.Module):

    def __init__(self, dim=192, scale=4, depth_kernel=3):
        super().__init__()

        scale_dim = dim * scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                     Rearrange('b n c -> b c n'),
                                     nn.BatchNorm1d(scale_dim),
                                     nn.GELU(),
                                     Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                     )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
            nn.BatchNorm2d(scale_dim),
            nn.GELU(),
            Rearrange('b c h w -> b (h w) c', h=14, w=14)
            )

        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                       Rearrange('b n c -> b c n'),
                                       nn.BatchNorm1d(dim),
                                       nn.GELU(),
                                       Rearrange('b c n -> b n c')
                                       )

    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x


class LCAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q[:, :, -1, :].unsqueeze(2)  # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, ch, useMulA=False, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            attennorm = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            if useMulA:
                attennorm = PreNorm(dim, MultiScaleAttention4D(d_input=ch, d_model=ch))
            self.layers.append(nn.ModuleList([
                attennorm,
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TransformerMA(nn.Module):
    def __init__(self, norm_shape, patchsize, d_in, d_out, d_mlp, deep=4, dropout=.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.norm = nn.LayerNorm(d_mlp)
        for _ in range(deep):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(norm_shape, MultiScaleAttention5D(patchsize=patchsize, d_input=d_in, d_model=d_out)),
                    FeedForward(d_in, d_mlp, dropout=dropout)
                ])
            )
        # self.conv = nn.Conv2d(in_channels=d_in, out_channels=d_in//2, kernel_size=1)

    def forward(self, x):
        for attn, ff in self.layers:
            _, t, _, _, _ = x.shape
            x = attn(x) + x
            x_ff = ff(rearrange(x, 'b t c h w -> b h w (t c)'))
            x_ff = rearrange(x_ff, 'b h w (t c) -> b t c h w', t=t)
            x += x_ff
        return x