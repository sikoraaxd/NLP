import torch
import torch.nn as nn
from model.transformer_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 forward_exxpansion, 
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(embed_size, heads, forward_exxpansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = dropout

    
    def forward(self, x, mask):
        N_words, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N_words, seq_length)

        position_encode = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
       
        out = position_encode
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out