import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.attention_norm = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)

        self.output_fc = nn.Sequentia(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, values, keys, query, mask):
        attention = self.attention(values, keys, query, mask)
        x = self.dropout(self.attention_norm(attention + query))
        
        forward = self.output_fc(x)
        out = self.dropout(self.output_norm(forward+x))

        return out



class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.encoder_block = EncoderBlock(embed_size, heads, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.encoder_block(value, key, query, src_mask)
        return out