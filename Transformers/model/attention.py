import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
            super(MultiHeadAttention, self).__init__()
            self.embed_size = embed_size
            self.heads = heads
            self.head_size = embed_size // heads

            self.keys = nn.Linear(self.head_size, self.head_size)
            self.values = nn.Linear(self.head_size, self.head_size)
            self.queries = nn.Linear(self.head_size, self.head_size)
            
            self.fc = nn.Linear(self.head_size*heads, embed_size)


    def forward(self, values, keys, query, mask):
        N_words = query.shape[0]

        values_size, keys_size, query_size = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N_words, values_size, self.heads, self.head_size)
        keys = keys.reshape(N_words, keys_size, self.heads, self.head_size)
        queries = query.reshape(N_words, query_size, self.heads, self.head_size)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        numerator = torch.einsum('nqhz, nkhz -> nhqk', [queries, keys])

        if mask is not None:
            numerator = numerator.masked_fill(mask==0, float("-inf"))

        attention = torch.softmax((numerator/torch.sqrt(query_size)), dim=-1)

        out = torch.einsum('nhqk,nkhz->nqhz', [attention, values]).reshape(
            N_words, query_size, self.heads*self.head_size)
        
        out = self.fc(out)
        return out