# %%
# rely
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse


# %%
# attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        p = torch.bmm(q, k.transpose(1, 2))
        p = p / (q.size(2) ** 0.5)

        if mask is not None:
            p = p.masked_fill(mask, -np.inf)

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

        self.linear_v = nn.Linear(d_model, heads * self.d_v)
        self.linear_k = nn.Linear(d_model, heads * self.d_v)
        self.linear_q = nn.Linear(d_model, heads * self.d_v)

        self.attn = ScaledDotProductAttention()
        self.linear = nn.Linear(heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        bs = q.size(0)
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(int(bs * self.heads), -1, self.d_v)
        k = k.view(int(bs * self.heads), -1, self.d_v)
        v = v.view(int(bs * self.heads), -1, self.d_v)

        if mask is not None:
            mask = mask.repeat(self.heads, 1, 1)

        at, op = self.attn(q, k, v, mask)
        op = op.view(bs, -1, self.d_v * self.heads)
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
        #o *= non_pad_mask
        o = self.ffn(o)
        #o *= non_pad_mask
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
        #o *= non_pad_mask

        en_at, o = self.encattn(o, en_o, en_o, mask=endn_mask)
        #o *= non_pad_mask

        o = self.ffn(o)
        #o *= non_pad_mask

        return o, at, en_at


# %%
# position encoding
def get_sin(len, word_d, d_model):
    def get_pow(pos, i):
        return pos / (10000 ** (2 * (i // 2) / d_model))

    def get_pos(x):
        return [get_pow(x, i) for i in range(word_d)]

    p = np.array([get_pos(i) for i in range(len)])
    p[:, 0::2] = np.sin(p[:, 0::2])
    p[:, 1::2] = np.cos(p[:, 1::2])

    return torch.FloatTensor(p)


# %%
# padding mask
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    mask = seq_k.eq(0)
    mask = mask.unsqueeze(1).expand(-1, len_q, -1)
    return mask


# %%
# sequence mask
def sequence_mask(seq):
    ba, lens = seq.size()
    mask = torch.triu(
        torch.ones((lens, lens), device=seq.device, dtype=torch.uint8), diagonal=1
    )
    mask = mask.unsqueeze(0).expand(ba, -1, -1)
    return mask


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

        slf_mask = padding_mask(seq, seq)
        o = self.input_emb(seq) + self.pos_emb(pos)
        for layer in self.layers:
            at, o = layer(o, slf_attn_mask=slf_mask)
            attn_list.append(at)

        #if return_attns:
        return at, o
        #return o,


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

        slf_mask = padding_mask(tgt_seq, tgt_seq)
        seq_mask = sequence_mask(tgt_seq)
        mask = (slf_mask.type(torch.uint8) + seq_mask.type(torch.uint8)).gt(0)
        con_mask = padding_mask(src_seq, tgt_seq)

        o = self.input_emb(tgt_seq) + self.pos_emb(tgt_pos)
        for layer in self.layers:
            o, at, en_at = layer(o, enc_output, slf_attn_mask=mask, endn_mask=con_mask)
            if return_attn:
                slf_attn.append(at)
                con_attn.append(en_at)

        #if return_attn:
        return o, slf_attn, con_attn
        #return o,


# %%
# transformer
class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, max_len, word_ved_d=512,
                 d_model=512, d_ffn=2048, n_layers=6, n_head=8, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(n_src_vocab, max_len, word_ved_d,
                               n_layers, n_head, d_model, d_ffn, dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab, max_len, word_ved_d,
                               n_layers, n_head, d_model, d_ffn, dropout=dropout)
        self.linear = nn.Linear(d_model, n_tgt_vocab)
        nn.init.xavier_normal_(self.linear.weight)
        self.linear.weight = self.decoder.input_emb.weight
        self.scale = (d_model ** -0.5)
        #self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        en_at, en_o = self.encoder(src_seq, src_pos)
        de_o, de_at, at = self.decoder(tgt_seq, tgt_pos, src_seq, en_o)
        de_o = self.linear(de_o) * self.scale
        #de_o = self.softmax(de_o)

        return de_o.view(-1, de_o.size(2))

# %%
# optimizer
class ScheduleOptim():
    def __init__(self, optim, d_model, n_warmup_steps):
        self._optimizer = optim
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
        ])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for para in self._optimizer.param_groups:
            para['lr'] = lr


# %%
# dataset
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_w2i, tgt_w2i, src_ins, tgt_ins):
        super().__init__()
        self._src_w2i = src_w2i
        self._src_i2w = {idx: word for word, idx in src_w2i.items()}
        self._src_ins = src_ins
        self._tgt_w2i = tgt_w2i
        self._tgt_i2w = {idx: word for word, idx in tgt_w2i.items()}
        self._tgt_ins = tgt_ins

    def __len__(self):
        return len(self._src_ins)

    def __getitem__(self, idx):
        if self._tgt_ins:
            return self._src_ins[idx], self._tgt_ins[idx]
        return self._src_ins[idx]


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)
    return (*src_inst, *tgt_inst)


def collate_fn(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [0] * (max_len - len(inst))
        for inst in insts
    ])
    batch_pos = np.array([
        [pos_i + 1 if w_i != 0 else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq
    ])
    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)
    return batch_seq, batch_pos


def cal_loss(pred, gold, smoothing=False):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')
    return loss


def cal_performance(pred, gold, smoothing=False):
    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(0)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def train_epoch(model, training_data, optimizer, device):
    model.train()
    t_loss = 0
    words = 0
    corr = 0
    for batch in tqdm(training_data, mininterval=2,
                      desc='-(Training)   ', leave=False):
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        loss, n_correct = cal_performance(pred, gold, smoothing=False)
        loss.backward()

        optimizer.step_and_update_lr()

        t_loss += loss.item()

        non_pad_mask = gold.ne(0)
        word = non_pad_mask.sum().item()
        words += word
        corr += n_correct

    loss_per_word = t_loss / words
    accuracy = corr / words
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    model.eval()
    t_loss = 0
    words = 0
    corr = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation)  ', leave=False):
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold)

            t_loss += loss.item()
            non_pad_mask = gold.ne(0)
            word = non_pad_mask.sum().item()
            words += word
            corr += n_correct

    loss_per_word = t_loss / words
    accuracy = corr / words
    return loss_per_word, accuracy


if __name__ == '__main__':
    data = torch.load('data/multi30k.atok.low.pt')
    train_data = torch.utils.data.DataLoader(
        TranslationDataset(data['dict']['src'], data['dict']['tgt'],
                           data['train']['src'], data['train']['tgt']),
        batch_size=64,
        collate_fn=paired_collate_fn
    )
    valid_data = torch.utils.data.DataLoader(
        TranslationDataset(data['dict']['src'], data['dict']['tgt'],
                           data['valid']['src'], data['valid']['tgt']),
        batch_size=64,
        collate_fn=paired_collate_fn
    )
    device = torch.device('cuda')
    transformer = Transformer(
        len(data['dict']['src']), len(data['dict']['tgt']),
        data['settings'].max_word_seq_len
    ).to(device)
    optimizer = ScheduleOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09
        ), 512, 4000
    )

    EPOCH = 100
    trlo = []
    trar = []
    valo = []
    vaar = []
    for i in range(EPOCH):
        print(str(i) + ":")
        train_loss, train_accu = train_epoch(transformer, train_data,
                                             optimizer, device)
        trlo.append(train_loss)
        trar.append(train_accu)
        valid_loss, valid_accu = eval_epoch(transformer, valid_data, device)
        valo.append(valid_loss)
        vaar.append(valid_accu)
        print(train_loss, train_accu, valid_loss, valid_accu)

    plt.plot(range(EPOCH), trlo)
    plt.plot(range(EPOCH), valo)
    plt.savefig('a.png')
    plt.clf()
    plt.plot(range(EPOCH), trar)
    plt.plot(range(EPOCH), vaar)
    plt.savefig('b.png')
    plt.clf()