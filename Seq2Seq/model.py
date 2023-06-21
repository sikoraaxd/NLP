import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)


    def forward(self, x):
        embbedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embbedding)

        return hidden, cell
        

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        preds = self.fc(outputs).squeeze(0)
        return preds, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    
    def forward(self, source, target, target_vocab, tfr=0.5):
        batch_size = source.shape[1]
        target_size = target.shape[0]
        target_vocab_size = len(target_vocab)

        outputs = torch.zeros(target_size, batch_size, target_vocab_size)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_size):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < tfr else best_guess
        
        return outputs