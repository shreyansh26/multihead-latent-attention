import torch
import torch.nn as nn
import math

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # print("Original freqs cis shape: ", freqs_cis.shape)
    # print(freqs_cis)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    # Keep shapes at seqlen, H/2 - (1, S, 1, H/2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(x, freqs_cis):
    # print("Original x szhape: ", x.shape)
    x_bsnh_2 = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # Group as complex nums pair-wise - B, S, Q, H/2
    # print("x shape: ", x_bsnh_2.shape)
    freqs_cis_1s1h_2 = reshape_for_broadcast(freqs_cis, x_bsnh_2)
    # print("freqs cis shape: ", freqs_cis.shape)
    # print("x * freqs cis shape: ", (x_bsnh_2 * freqs_cis_1s1h_2).shape)
    # print("x * freqs cis: ", x_bsnh_2 * freqs_cis_1s1h_2)
    # print("torch.view_as_real(x * freqs cis) shape: ", torch.view_as_real(x_bsnh_2 * freqs_cis_1s1h_2).shape)
    # print("torch.view_as_real(x * freqs cis): ", torch.view_as_real(x_bsnh_2 * freqs_cis_1s1h_2))
    x_out_bsqh_2 = torch.view_as_real(x_bsnh_2 * freqs_cis_1s1h_2).flatten(3)
    # print("x_out shape: ", x_out_bsqh_2.shape)
    # print("x_out: ", x_out_bsqh_2)
    return x_out_bsqh_2.type_as(x)


if __name__ == "__main__":
    xq = torch.randn(5, 1, 16, 128)
    xk = torch.randn(5, 100, 16, 128)
    freqs_cis_sh = precompute_freqs_cis(128, 100)
    xq_out = apply_rotary_emb(xq, freqs_cis_sh)
    xk_out = apply_rotary_emb(xk, freqs_cis_sh)
    print(xq_out.shape)
    print(xk_out.shape)