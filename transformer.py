# %%
# rely
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# %%
# attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = torch.dropout(dropout)
        self.softmax = torch.softmax()

    def forward(self, q, k, v, mask=None):
        p = torch.bmm(q, k.transpose(1, 2))
        p = p / (q.size(2) ** 0.5)

        if mask is not None:
            p = p.masked_fill(mask, -torch.inf)

        o = self.softmax(p)
        o = self.dropout(o)
        o = torch.bmm(o, v)

        return p, o


# %%
# multihead
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_v = d_model // heads

        self.linear_v = nn.Linear(d_model, heads*self.d_v)
        self.linear_k = nn.Linear(d_model, heads*self.d_v)
        self.linear_q = nn.Linear(d_model, heads*self.d_v)

        self.attn = ScaledDotProductAttention()
        self.linear = nn.Linear(heads*self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        bs = q.size(0)
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q.view(bs*self.heads, -1, self.d_v)
        k.view(bs*self.heads, -1, self.d_v)
        v.view(bs*self.heads, -1, self.d_v)

        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        at, op = self.attn(q, k, v, mask)
        op.view(bs, -1, self.d_v*self.heads)
        op = self.linear(op)
        op = self.dropout(op)
        op = self.layernorm(op + residual)

        return at, op


# %%
# position wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.w1 = nn.Conv1d(d_model, d_ffn, 1)
        self.w2 = nn.Conv1d(d_ffn, d_model, 1)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        o = x.transpose(1, 2)
        o = self.w2(F.relu(self.w1(o)))
        o = o.transpose(1, 2)
        o = self.dropout(o)
        o = self.layernorm(o + residual)
        return o


# %%
# encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ffn, dropout=dropout)

    def forward(self, x, non_pad_mask=None, slf_attn_mask=None):
        at, o = self.attn(x, x, x, mask=slf_attn_mask)
        o *= non_pad_mask
        o = self.ffn(o)
        o *= non_pad_mask
        return at, o


# %%
# decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.slfattn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.encattn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ffn, dropout=dropout)

    def forward(self, x, en_o, non_pad_mask=None, slf_attn_mask=None, endn_mask=None):
        at, o = self.slfattn(x, x, x, mask=slf_attn_mask)
        o *= non_pad_mask

        en_at, o = self.encattn(o, en_o, en_o, mask= endn_mask)
        o *= non_pad_mask

        o = self.ffn(o)
        o *= non_pad_mask

        return o, at, en_at


# %%
# position encoding
def get_sin(len, word_d, d_model):

    def get_pow(pos, i):
        return pos / (10000 ** (2 * (i//2) / d_model))

    def get_pos(x):
        return [get_pow(x, i) for i in range(word_d)]

    p = np.array([get_pos(i) for i in range(len)])
    p[:, 0::2] = np.sin(p[:, 0::2])
    p[:, 1::2] = np.cos(p[:, 1::2])

    return torch.FloatTensor(p)

# %%
# attention_key_pad_mask

# %%
# encoder
class Encoder(nn.Module):
    def __init__(self, n_src_vocab, max_len, word_vec_d,
                 n_layers, n_head, d_model, d_ffn, dropout=0.1):
        super().__init__()
        n_position = max_len + 1
        self.input_emb = nn.Embedding(n_src_vocab, word_vec_d)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin(n_position, word_vec_d, d_model)
        )
        self.layers = nn.ModuleList([
            EncoderLayer(n_head, d_model, d_ffn, dropout=dropout) for i in range(n_layers)
        ])

    def forward(self, seq, pos, return_attns=False):
        attn_list = []

        o = self.input_emb(seq) + self.pos_emb(pos)
        for layer in self.layers:
            at, o = layer(o)
            attn_list.append(at)

        if return_attns:
            return at, o
        return o


# %%
# decoder
class Decoder(nn.Module):
    def __init__(self, n_tgt_vocab, max_len, word_vec_d,
                 n_layers, n_head, d_model, d_ffn, dropout=0.1):
        super().__init__()
        n_position = max_len + 1
        self.input_emb = nn.Embedding(n_tgt_vocab, word_vec_d)
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sin(n_position, word_vec_d, d_model)
        )
        self.layers = nn.ModuleList([
            DecoderLayer(n_head, d_model, d_ffn, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attn=False):
        slf_attn = []
        con_attn = []

        o = self.input_emb(tgt_seq) + self.pos_emb(tgt_pos)
        for layer in self.layers:
            o, at, en_at = layer(o, enc_output)
            if return_attn:
                slf_attn.append(at)
                con_attn.append(en_at)

        if return_attn:
            return o, slf_attn, con_attn
        return o
# %%
# transformer
class Transformer(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass








