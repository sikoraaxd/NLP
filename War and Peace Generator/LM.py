import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


BATCH_SIZE = 16
EMBED_SIZE = 510
BLOCK_SIZE = 128
HEAD_SIZE = 6
EPOCHS = 5000
EPOCHS_VAL = 200
N_GRAMM = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('./war_and_peace.txt', 'r', encoding='utf-8') as f:
        text = f.read()

def get_ngramms(n, text):
    vocab = set([])
    for i in range(len(text)-n):
        vocab.add(text[i:i+n])
    
    return sorted(list(vocab))


ngramms = get_ngramms(n=N_GRAMM, text=text)
VOCAB_SIZE = len(ngramms)

stoi = {ch:i for i, ch in enumerate(ngramms)}
itos = {i:ch for i, ch in enumerate(ngramms)}  

def encode(s):
    result = []
    for i in range(0, len(s)-N_GRAMM, N_GRAMM):
        result.append(stoi[s[i:i+N_GRAMM]])
    return result

decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9*len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]


def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EPOCHS_VAL)
        for k in range(EPOCHS_VAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(0.2)

    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(1, 2) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        out = w @ v
        return out
    


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(EMBED_SIZE, EMBED_SIZE)
        self.dropout = nn.Dropout(0.2)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))



class FFN(nn.Module):
    def __init__(self, embed_size):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size),
            nn.Dropout(0.2)
        )

    
    def forward(self, x):
        return self.ffn(x)
    


class Block(nn.Module):
    def __init__(self, embed_size, head_size):
        super(Block, self).__init__()
        h_size = embed_size // head_size
        self.sa = MultiHeadAttention(head_size, h_size)
        self.ffn = FFN(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return(x)



class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)
        self.blocks = nn.Sequential(*[Block(EMBED_SIZE, HEAD_SIZE) for _ in range(6)])
        self.ln = nn.LayerNorm(EMBED_SIZE)
        self.lm_head = nn.Linear(EMBED_SIZE, VOCAB_SIZE)
        

    def forward(self, x, y=None):
        B, T = x.shape
        tok_embed = self.token_embed(x)
        pos_embed = self.pos_embed(torch.arange(T))
        out = tok_embed + pos_embed
        out = self.blocks(out)
        out = self.ln(out)
        out = self.lm_head(out)
        
        if y is None:
            loss = None
        else:
            B, T, C = out.shape
            out = out.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(out, y)

        return out, loss


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 

    
    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            x_cond = x[:, -BLOCK_SIZE:]
            out, loss = self(x_cond)
            out = out[:, -1, :]
            probs = F.softmax(out, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x


model = LM()
best_model = None
best_loss = float('inf')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for epoch in tqdm(range(EPOCHS)):
    xb, yb = get_batch('train')

    out, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_loss:
            best_loss = losses['val']
            best_model = model.state_dict()
        
        x = torch.zeros((1, 1), dtype=torch.long)
        generated_text = decode(model.generate(x, max_new_tokens=10000)[0].tolist())
        with open(f'step_{epoch}.txt', 'w', encoding='utf-8') as f:
            f.write(generated_text)

torch.save(best_model, 'war_and_peace_transformer.pth')