import spacy
import torch
import torch.nn as nn
import torchdata.datapipes as dp
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator
from model import Decoder, Encoder, Seq2Seq
import warnings 
import tqdm
import boto3


warnings.filterwarnings('ignore')

TRAIN_SIZE = 0.6
LR = 0.001
EPOCHS = 30
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_ru = spacy.load("ru_core_news_lg")
spacy_eng = spacy.load('en_core_web_trf')

def tokenizer_ru(text):
    return [tok.text for tok in spacy_ru.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def getTokens(data_iter, place):
    for russian, english in data_iter:
        if place == 0:
            yield tokenizer_ru(russian)
        else:
            yield tokenizer_eng(english)


def getTransform(vocab):
    text_tranform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.AddToken(1, begin=True),
        T.AddToken(2, begin=False)
    )
    return text_tranform

def save_data(filename):
    s3 = boto3.client('s3',
                        endpoint_url='https://storage.yandexcloud.net',
                        aws_access_key_id='YCAJEffTHMPCfFn4jBYUDB6oV',
                        aws_secret_access_key='YCNdxgyAtr7bUzU0iIQeQi9ViIJ7GS-ZdbiR3Fyo')

    bucket_name = 'sikoraaxd-bucket'

    with open(filename, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, filename)


FILE_PATH = "./ru_eng.csv"
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=1, delimiter='\t', as_tuple=True)

source_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,0),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

target_vocab = build_vocab_from_iterator(
    getTokens(data_pipe,1),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
target_vocab.set_default_index(target_vocab['<unk>'])

def applyTransform(sequence_pair):
    return (
        getTransform(source_vocab)(tokenizer_ru(sequence_pair[0])),
        getTransform(target_vocab)(tokenizer_eng(sequence_pair[1]))
    )


def sortBucket(bucket):
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))


def separateSourceTarget(sequence_pairs):
    sources,targets = zip(*sequence_pairs)
    return sources,targets


def applyPadding(pair_of_sequences):
    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))

data_pipe = data_pipe.map(applyTransform)

data_pipe = data_pipe.bucketbatch(
    batch_size = BATCH_SIZE,  bucket_num=1,
    use_in_batch_shuffle=False, sort_key=sortBucket
)

data_pipe = data_pipe.map(separateSourceTarget)
data_pipe = data_pipe.map(applyPadding)

split_idx = int(len(list(data_pipe))*TRAIN_SIZE)
train_pipe = list(data_pipe)[:split_idx]
test_pipe = list(data_pipe)[split_idx:]

input_size_encoder = len(source_vocab)
input_size_decoder = len(target_vocab)
output_size = len(target_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.4
decoder_dropout = 0.4
pad_idx = source_vocab.get_stoi()['<pad>']

encoder = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

state_checkpoints = []

for epoch in range(EPOCHS):
    for sources, targets in tqdm.tqdm(train_pipe):
        sources = sources.T.to(device)
        targets = targets.T.to(device)
        output = model(sources, targets, target_vocab)

        output = output[1:].reshape(-1, output.shape[2])
        target = targets[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
    
    print(f'Epoch: {epoch}, loss: {loss.item()}')
    
    state_checkpoints.append({
        'epoch': epoch,
        'loss': loss.item(),
        'state_dict': model.state_dict()
    })

torch.save(model.cpu().state_dict(), 'ru_eng_seq2seq.pth')
save_data('ru_eng_seq2seq.pth')