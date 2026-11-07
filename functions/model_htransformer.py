from math import log2, ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from einops import rearrange, repeat


def exists(val):
    return val is not None


def masked_aggregate(tensor, mask = None, dim = -1, average = True):
    if not exists(mask):
        return torch.mean(tensor, dim = dim) if average else torch.sum(tensor, dim = dim)
    diff = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff))]
    tensor = tensor.masked_fill(~mask, 0.)
    total_el = mask.sum(dim = dim)
    agg = tensor.sum(dim = dim)
    if average:
        agg = agg / total_el.clamp(min = 1.)
    return agg.masked_fill(total_el == 0, 0.)


def shift(t, amount, mask = None):
    if amount == 0:
        return t
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)
    return F.pad(t, (0, 0, amount, -amount), value = 0.)


class SequentialSequence(nn.Module):
    """Sequential transformer execution maintaining checkpoint compatibility."""
    def __init__(self, layers, args_route = None, layer_dropout = 0.0):
        super().__init__()
        self.layers = layers
        self.args_route = args_route or {}
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        for attn, ff in self.layers:
            x = x + attn(x, mask = mask)
            x = x + ff(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, *, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    
    def forward(self, x):
        return self.net(x)


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)
    
    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        feats_per_shift = x.shape[-1] // len(self.shifts)
        splitted = x.split(feats_per_shift, dim = -1)
        shifted = [shift(s, sh, mask = mask) for s, sh in zip(splitted[:len(self.shifts)], self.shifts)]
        return self.fn(torch.cat((*shifted, *splitted[len(self.shifts):]), dim = -1), **kwargs)


def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))
    return rearrange(t, 'b n r ... -> b (n r) ...')


class HAttention1D(nn.Module):
    """Hierarchical 1D Multi-Head Attention."""
    def __init__(self, dim, *, heads = 20, dim_head = 64, block_size = 16, pos_emb = None, eps = 1e-8, **kwargs):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head
        self.pos_emb = pos_emb
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x, mask = None):
        b, n, h, device, bsz, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.eps
        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n
        
        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)
            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)
        
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        if exists(mask):
            mask = repeat(mask, 'b n -> (b h) n', h = h)
        
        q = q * self.scale
        if exists(self.pos_emb):
            freqs = rearrange(self.pos_emb(torch.arange(pad_to_len, device = device)), 'n d -> () n d')
            q, k, v = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        
        num_levels = int(log2(pad_to_len // bsz)) - 2
        qkvs = [(q, k, v, mask)]
        for _ in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))
            if exists(mask):
                mask = repeat(mask, 'b (n r) -> b n r', r = 2)
            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)
            if exists(mask):
                mask = torch.any(mask, dim = 2)
            qkvs.append((q, k, v, mask))
        qkvs = [qkvs[0], *qkvs]
        
        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)
            if exists(mask):
                S = S.masked_fill(~mask, -torch.finfo(S.dtype).max)
            S = S - torch.max(S, dim = -1, keepdim = True).values
            A = S.exp()
            y = einsum('... i j, ... j d -> ... i d', A, v)
            return rearrange(y, 'b ... n d -> b (... n) d'), rearrange(A.sum(dim = -1), 'b ... i -> b (... i)')
        
        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)
        Ys = []
        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = (ind == len(qkvs) - 1)
            q, k, v = map(to_blocks, (q, k, v))
            S_mask = None
            if exists(mask):
                mask = to_blocks(mask)
                k_mask = flip_every_two(mask.int()).bool() if not is_last else mask
                S_mask = rearrange(mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')
            if not is_last:
                k, v = map(flip_every_two, (k, v))
            Ys.append(calculate_Y_and_A(q, k, v, mask = S_mask))
        
        Y, A = 0, 0
        for ind, (Y_level, A_level) in enumerate(Ys):
            is_last = (ind == len(Ys) - 1)
            if not is_last and torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r = 2)
                A = repeat(A, 'b n -> b (n r)', r = 2)
            Y = Y_level + Y
            A = A_level + A
        
        out = Y / rearrange(A + eps, 'b n -> b n ()')
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out[:, :n])


class HTransformer1D(nn.Module):
    """
    Unified Hierarchical 1D Transformer supporting CLS token and Average pooling
    with multimodal tabular feature fusion.
    """
    def __init__(
            self,
            *,
            num_classes   : int  = 2,
            num_tokens    : int  = 5000,
            embedding_dim : int  = 512,
            max_seq_len   : int  = 8427,
            depth         : int  = 3,
            heads         : int  = 4,
            dim_head      : int  = 32,
            ff_mult       : int  = 4,
            block_size    : int  = 64,
            pooling       : str  = 'cls',
            shift_tokens  : bool = False,
            **kwargs
        ):
        super().__init__()
        assert (max_seq_len % block_size) == 0, 'max_seq_len must be divisible by block_size'
        num_blocks = max_seq_len // block_size
        assert log2(num_blocks).is_integer(), f'number of blocks {num_blocks} must be a power of 2'
        
        self.pooling = pooling
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.pos_emb = RotaryEmbedding(dim = dim_head)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) if pooling == 'cls' else None
        
        layers = nn.ModuleList([])
        shift_ranges = (0, 1) if shift_tokens else (-1, 0, 1)
        for _ in range(depth):
            attn = HAttention1D(embedding_dim, dim_head = dim_head, heads = heads, block_size = block_size, pos_emb = self.pos_emb)
            ff = FeedForward(embedding_dim, mult = ff_mult)
            if shift_tokens:
                attn, ff = PreShiftTokens(shift_ranges, attn), PreShiftTokens(shift_ranges, ff)
            layers.append(nn.ModuleList([PreNorm(embedding_dim, attn), PreNorm(embedding_dim, ff)]))
        
        self.layers = SequentialSequence(layers, args_route = {'mask': ((True, False),) * depth})
        self.atc_norm = nn.LazyBatchNorm1d()
        
        # Classification head
        self.mlp = nn.Sequential(
            nn.LazyLinear(embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(embedding_dim // 4),
            nn.BatchNorm1d(embedding_dim // 4),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, tokens, data, masks = None, mask = None):
        attn_mask = masks if masks is not None else mask
        b = tokens.shape[0]
        embeddings = self.token_emb(tokens)
        
        if self.pooling == 'cls':
            embeddings = torch.cat([self.cls_token.expand(b, -1, -1), embeddings], dim = 1)
            if exists(attn_mask):
                attn_mask = torch.cat([torch.ones((b, 1), dtype = torch.bool, device = tokens.device), attn_mask], dim = 1)
                
        attn_out = self.layers(embeddings, mask = attn_mask)
        pooled = attn_out[:, 0, :] if self.pooling == 'cls' else torch.mean(attn_out, dim = 1)
        return self.mlp(torch.cat([pooled, data], dim = 1).float())


class HTransformer1D_custom_clsPooling(HTransformer1D):
    """CLS-pooling variant of HTransformer1D."""
    def __init__(self, *args, **kwargs):
        kwargs['pooling'] = 'cls'
        super().__init__(*args, **kwargs)


class HTransformer1D_custom_averagePooling(HTransformer1D):
    """Average-pooling variant of HTransformer1D."""
    def __init__(self, *args, **kwargs):
        kwargs['pooling'] = 'avg'
        super().__init__(*args, **kwargs)
