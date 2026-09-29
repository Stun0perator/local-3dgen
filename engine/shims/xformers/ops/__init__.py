"""memory_efficient_attention and fmha.BlockDiagonalMask, implemented with torch SDPA.

Layout matches xformers: q/k/v are [B, N, H, C]; output is [B, N, H, C].
"""
import torch
import torch.nn.functional as F

from . import fmha


def _sdpa(q, k, v, mask=None):
    # [B, N, H, C] -> [B, H, N, C]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         attn_mask=mask)
    return out.transpose(1, 2)


def memory_efficient_attention(q, k, v, attn_bias=None):
    if attn_bias is None:
        return _sdpa(q, k, v)
    if not isinstance(attn_bias, fmha.BlockDiagonalMask):
        raise NotImplementedError("only BlockDiagonalMask is supported")
    # Variable-length segments packed along N (B == 1). Pad segments of similar length into
    # batches so short windows don't pay for the longest one.
    q_lens, kv_lens = attn_bias.q_seqlen, attn_bias.kv_seqlen
    q_off = [0]; kv_off = [0]
    for a, b in zip(q_lens, kv_lens):
        q_off.append(q_off[-1] + a); kv_off.append(kv_off[-1] + b)
    out = torch.empty_like(q)
    groups = {}
    for i, (a, b) in enumerate(zip(q_lens, kv_lens)):
        groups.setdefault((a, b), []).append(i)
    for (a, b), idx in groups.items():
        for s in range(0, len(idx), 256):
            chunk = idx[s:s + 256]
            qs = torch.stack([q[0, q_off[i]:q_off[i] + a] for i in chunk])
            ks = torch.stack([k[0, kv_off[i]:kv_off[i] + b] for i in chunk])
            vs = torch.stack([v[0, kv_off[i]:kv_off[i] + b] for i in chunk])
            o = _sdpa(qs, ks, vs)
            for j, i in enumerate(chunk):
                out[0, q_off[i]:q_off[i] + a] = o[j]
    return out
