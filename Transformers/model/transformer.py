import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 target_vocab_size,
                 src_pad_index,
                 target_pad_index,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=2,
                 heads=8,
                 dropout=0.1,
                 max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, 
                               embed_size, 
                               num_layers, 
                               heads, 
                               forward_expansion,
                               dropout,
                               max_length)
        
        self.decoder = Decoder(target_vocab_size,
                               embed_size, 
                               num_layers, 
                               heads, 
                               forward_expansion,
                               dropout,
                               max_length)
        
        self.src_pad_index = src_pad_index
        self.target_pad_index = target_pad_index
        
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)
        return src_mask
    

    def make_target_mask(self, target):
        N, target_length = target.shape
        target_mask = torch.tril(torch.ones((target_length, target_length))).expand(
            N, 1, target_length, target_length
        )

        return target_mask
    

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)

        return out