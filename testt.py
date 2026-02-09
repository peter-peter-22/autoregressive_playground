import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size = 65
batch_size = 32
seq_len = 64
d_model = 128

device="cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

emb = nn.Embedding(vocab_size, d_model).to(device)
head = nn.Linear(d_model, vocab_size).to(device)

opt = torch.optim.AdamW(
    list(emb.parameters()) + list(head.parameters()),
    lr=1e-3
)

step=0
while True:
    opt.zero_grad()
    h = emb(x)
    logits = head(h)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )
    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(f"step {step}, loss {loss.item():.4f}")

    if torch.isnan(loss):
        print("NaN at step", step)
        break
    step+=1
