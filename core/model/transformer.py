from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):

        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])


class TokenEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, trg_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.trg_tok_emb = TokenEmbedding(trg_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)


    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_padding_mask,
                trg_padding_mask):
        
        src_length = src.size(1)
        trg_length = trg.size(1)

        src_mask = torch.zeros((src_length, src_length),device=src.device).type(torch.bool)

        trg_mask = self.generate_square_subsequent_mask(trg_length, trg.device)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        trg_emb = self.positional_encoding(self.trg_tok_emb(trg))
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask = src_padding_mask)
        return torch.log_softmax(self.generator(outs), dim=-1)


    def encode(self, src: Tensor, src_key_padding_mask):

        src_length = src.size(1)
        src_mask = torch.zeros((src_length, src_length),device=src.device).type(torch.bool)

        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_key_padding_mask)

    def decode(self, trg: Tensor, memory: Tensor, pad_symbol: int, memory_key_padding_mask = None):
        
        trg_length = trg.size(1)

        trg_mask = self.generate_square_subsequent_mask(trg_length, trg.device)

        trg_padding_mask = (trg == pad_symbol)

        return self.transformer.decoder(self.positional_encoding(
                                            self.trg_tok_emb(trg)),
                                            memory,
                                            trg_mask,
                                            memory_key_padding_mask=memory_key_padding_mask,
                                            tgt_key_padding_mask = trg_padding_mask)
    
    def generate(self, src, src_key_padding_mask, start_symbol, end_symbol, pad_symbol, max_len=100):
        DEVICE = src.device

        bz = src.size(0)

        memory = self.encode(src, src_key_padding_mask)

        ys = torch.ones(bz, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

        for i in range(max_len):
            memory = memory.to(DEVICE)

            out = self.decode(ys, memory, pad_symbol, src_key_padding_mask)

            prob = self.generator(out[:, -1])

            next_word = torch.argmax(prob, dim=-1).view(bz,-1)

            ys = torch.cat([ys, next_word], dim=1)

            if torch.any(ys == end_symbol, dim=1).sum() == bz:
                break
        return ys

    def generate_square_subsequent_mask(self, sz, device="cuda"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

