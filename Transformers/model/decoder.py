import torch
import torch.nn as nn

from model.transformer_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, 
                 target_vocab_size, 
                 embed_size, 
                 num_layers, 
                 heads, 
                 forward_expansion, 
                 dropout, 
                 max_length):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_fc = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, encoder_out, src_mask, target_mask):
        N_words, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N_words, seq_length)
        
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)
        
        out = self.output_fc(x)
        
                         