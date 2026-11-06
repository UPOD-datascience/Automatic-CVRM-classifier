from math import log2, ceil
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from h_transformer_1d.reversible import ReversibleSequence, SequentialSequence
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from einops import rearrange, repeat


# helpers
def exists(val):
    return val is not None

def masked_aggregate(tensor, mask = None, dim = -1, average = True):
    if not exists(mask):
        fn = torch.sum if not average else torch.mean
        return fn(tensor, dim = dim)
    
    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor = tensor.masked_fill(~mask, 0.)
    
    total_el = mask.sum(dim = dim)
    agg = tensor.sum(dim = dim)
    
    if average:
        agg = agg / total_el.clamp(min = 1.)
    
    agg.masked_fill_(total_el == 0, 0.)
    return agg

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)
    
    return F.pad(t, (0, 0, amount, -amount), value = 0.)
#########

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            *,
            mult = 4
        ):
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    
    def forward(self, x):
        return self.net(x)
#########

# token shifting
class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)
    
    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)


# hierarchical attention helper functions
def cast_for_op(cast_type, fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        orig_type = t.dtype
        t = t.type(cast_type)
        out = fn(t, *args, **kwargs)
        out = out.type(orig_type)
        return out
    return inner

def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
    t = rearrange(t, 'b n r ... -> b (n r) ...')
    return t
#########

# attention
class HAttention1D(nn.Module):
    def __init__(
            self,
            dim,
            *,
            heads = 20,
            dim_head = 64,
            block_size = 16,
            pos_emb = None,
            eps = 1e-8,
            **kwargs
        ):
        
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
        
        # pad sequence length to power of 2
        
        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n
        
        #print(x.shape, padding, pad_to_len, n)
        
        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)
            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)
        
        # derive queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        
        # split out heads, and also divide sequence into blocks
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        
        if exists(mask):
            mask = repeat(mask, 'b n -> (b h) n', h = h)
        
        # scale
        q = q * self.scale
        
        # rotary pos emb
        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(pad_to_len, device = device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q, k, v = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        
        # calculate number of levels until 2 x 2
        num_levels = int(log2(pad_to_len // bsz)) - 2
        assert num_levels >= 0, 'number of levels must be at least greater than 0'
        
        # coarsening
        qkvs = [(q, k, v, mask)]
        
        for _ in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))
            
            if exists(mask):
                mask = repeat(mask, 'b (n r) -> b n r', r = 2)
            
            # masked mean for queries and keys, but not values
            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)
            
            if exists(mask):
                mask = torch.any(mask, dim = 2)
            
            coarsened_qkvs = (q, k, v, mask)
            qkvs.append(coarsened_qkvs)
        
        qkvs = [qkvs[0], *qkvs]  # duplicate the finest resolution an extra time, for the base diagonal
        
        # half-attention function
        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)
            
            if exists(mask):
                mask_value = -torch.finfo(S.dtype).max
                S = S.masked_fill(~mask, mask_value)
            
            S = S - torch.max(S, dim = -1, keepdim = True).values
            A = S.exp()
            
            y = einsum('... i j, ... j d -> ... i d', A, v)
            
            A = A.sum(dim = -1)
            
            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            
            return y, A
        
        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)
        
        # calculate Ys, as in the paper
        Ys = []
        
        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)
            
            q, k, v = map(to_blocks, (q, k, v))
            
            # generate the mask for S
            S_mask = None
            if exists(mask):
                mask = to_blocks(mask)
                q_mask = mask
                k_mask = cast_for_op(torch.int, flip_every_two)(mask) if not is_last else mask
                S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')
            
            # flip keys and values to capture the off-diagonals
            if not is_last:
                k, v = map(flip_every_two, (k, v))
            
            Y_level = calculate_Y_and_A(q, k, v, mask = S_mask)
            Ys.append(Y_level)
        
        # interpolate
        Y = 0
        A = 0
        
        for ind, (Y_level, A_level) in enumerate(Ys):
            is_last = ind == (len(Ys) - 1)
            
            if not is_last and torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r = 2)
            
            if not is_last and torch.is_tensor(A):
                A = repeat(A, 'b n -> b (n r)', r = 2)
            
            Y = Y_level + Y
            A = A_level + A
        
        out = Y / rearrange(A + eps, 'b n -> b n ()')
        
        # merge heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        
        # combine out
        return self.to_out(out[:, :n])


# Full Encoder
class HTransformer1D_custom_averagePooling(nn.Module):
    def __init__(
            self,
            *,
            num_classes     : int   = 2,
            num_tokens      : int   = 5000,
            embedding_dim   : int   = 512,
            max_seq_len     : int   = 8427,
            depth           : int   = 3,
            heads           : int   = 4,
            dim_head        : int   = 32,
            ff_mult         : int   = 4,
            block_size      : int   = 64,     # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
            reversible      : bool  = False,
            shift_tokens    : bool  = False
        ):
        
        super().__init__()
        
        assert (max_seq_len % block_size) == 0, 'maximum sequence length must be divisible by the block size'
        
        num_blocks = max_seq_len // block_size
        assert log2(max_seq_len // block_size).is_integer(), f'number of blocks {num_blocks} must be a power of 2'
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.token_emb = nn.Embedding(
            num_embeddings = num_tokens,
            embedding_dim = embedding_dim
        )
        
        self.pos_emb = RotaryEmbedding(dim = dim_head)
        
        self.max_seq_len = max_seq_len
        
        layers = nn.ModuleList([])
        
        # Creazione del riferimento alla classe HAttention1D
        attn_class = HAttention1D
        attn_kwargs = dict()
        
        shift_token_ranges = (0, 1) if shift_tokens else (-1, 0, 1)
        
        for _ in range(depth):
            attn = attn_class(
                dim = embedding_dim,
                dim_head = dim_head,
                heads = heads, 
                block_size = block_size,
                pos_emb = self.pos_emb,
                **attn_kwargs
            )
            
            ff = FeedForward(embedding_dim, mult = ff_mult)
            
            if shift_tokens:
                attn, ff = map(lambda t: PreShiftTokens(shift_token_ranges, t), (attn, ff))
            
            attn, ff = map(lambda t: PreNorm(embedding_dim, t), (attn, ff))
            
            layers.append(nn.ModuleList([attn, ff]))
        
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        
        self.layers = execute_type(layers, args_route = {**attn_route_map})
        
        self.atc_norm = nn.LazyBatchNorm1d()
        
        # Classification head
        self.mlp = nn.Sequential(
            nn.LazyLinear(self.embedding_dim//2),
            nn.BatchNorm1d(self.embedding_dim//2),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(self.embedding_dim//4),
            nn.BatchNorm1d(self.embedding_dim//4),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(self.num_classes)
        )
    
    def forward(
            self, 
            tokens,
            data, 
            mask = None
        ):
        
        b, n = tokens.shape
        assert n <= self.max_seq_len, 'sequence length must be less than the maximum sequence length'
        
        #print(tokens.shape)
        
        # Tokens embedding
        embeddings = self.token_emb(tokens)
        
        #print(embeddings.shape)
        
        # Multi-head Attention
        attn_batch = self.layers(embeddings, mask = mask)
        
        #print(attn_batch.shape)
        
        pooled_outputs = torch.mean(attn_batch, dim=1)
        
        #print(pooled_outputs.shape)
        
        '''
        # Isolate the cls tokens
        cls_batch = attn_batch[:, 0, :]
        
        #print(f'cls: {cls_batch.shape}')
        '''
        
        # Late Fusion
        embedding_total = torch.cat([pooled_outputs, data], dim=1)
        
        #print(embedding_total.shape)
        
        # Outputs from the classification head
        outputs = self.mlp(embedding_total.float())
        
        return outputs


class HTransformer1D_custom_clsPooling(nn.Module):
    def __init__(
            self,
            *,
            num_classes     : int   = 2,
            num_tokens      : int   = 5000,
            embedding_dim   : int   = 512,
            max_seq_len     : int   = 8427,
            depth           : int   = 3,
            heads           : int   = 4,
            device          : str   = 'cuda',
            dim_head        : int   = 32,
            ff_mult         : int   = 4,
            block_size      : int   = 64,     # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
            reversible      : bool  = False,
            shift_tokens    : bool  = False
        ):
        
        super().__init__()
        
        assert (max_seq_len % block_size) == 0, 'maximum sequence length must be divisible by the block size'
        
        num_blocks = max_seq_len // block_size
        assert log2(max_seq_len // block_size).is_integer(), f'number of blocks {num_blocks} must be a power of 2'
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.token_emb = nn.Embedding(
            num_embeddings = num_tokens,
            embedding_dim = embedding_dim
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.pos_emb = RotaryEmbedding(dim = dim_head)
        
        self.max_seq_len = max_seq_len
        
        self.device = device
        
        layers = nn.ModuleList([])
        
        # Creazione del riferimento alla classe HAttention1D
        attn_class = HAttention1D
        attn_kwargs = dict()
        
        shift_token_ranges = (0, 1) if shift_tokens else (-1, 0, 1)
        
        for _ in range(depth):
            attn = attn_class(
                dim = embedding_dim,
                dim_head = dim_head,
                heads = heads, 
                block_size = block_size,
                pos_emb = self.pos_emb,
                **attn_kwargs
            )
            
            ff = FeedForward(embedding_dim, mult = ff_mult)
            
            if shift_tokens:
                attn, ff = map(lambda t: PreShiftTokens(shift_token_ranges, t), (attn, ff))
            
            attn, ff = map(lambda t: PreNorm(embedding_dim, t), (attn, ff))
            
            layers.append(nn.ModuleList([attn, ff]))
        
        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        
        self.layers = execute_type(layers, args_route = {**attn_route_map})
        
        self.atc_norm = nn.LazyBatchNorm1d()
        
        # Classification head
        self.mlp = nn.Sequential(
            nn.LazyLinear(self.embedding_dim//2),
            nn.BatchNorm1d(self.embedding_dim//2),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(self.embedding_dim//4),
            nn.BatchNorm1d(self.embedding_dim//4),
            nn.Dropout1d(p = 0.2),
            nn.ReLU(),
            nn.LazyLinear(self.num_classes)
        )
    
    def forward(
            self, 
            tokens,
            data, 
            masks = None
        ):
        
        b, n = tokens.shape
        assert n <= self.max_seq_len, 'sequence length must be less than the maximum sequence length'
        
        #print(tokens.shape)
        
        # Tokens embedding
        embeddings = self.token_emb(tokens)
        
        cls_masks = repeat(
            torch.tensor([[1]],dtype=torch.bool),
            '1 d -> b d',
            b = b
        ).to(tokens.device)
        
        masks = torch.cat((cls_masks, masks), dim=1)
        
        #print(cls_masks.shape, masks.shape)
        
        #######################################
        self.cls_tokens = repeat(
            self.cls_token,
            '1 1 d -> b 1 d',
            b = b
        )
        
        cls_embeddings = torch.cat((self.cls_tokens, embeddings), dim = 1)
        ##########################################
        
        #print(cls_embeddings.shape)
        
        #print(embeddings.shape)
        
        # Multi-head Attention
        attn_batch = self.layers(cls_embeddings, mask = masks)
        
        #print(attn_batch.shape)
        
        # Isolate the cls tokens
        cls_batch = attn_batch[:, 0, :]
        
        #print(cls_batch.shape, data.shape)
        
        # Late Fusion
        embedding_total = torch.cat([cls_batch, data], dim=1)
        
        #print(embedding_total.shape)
        
        # Outputs from the classification head
        outputs = self.mlp(embedding_total.float())
        
        return outputs