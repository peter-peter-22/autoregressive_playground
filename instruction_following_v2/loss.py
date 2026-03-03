import torch
import torch.nn as nn
from millify import millify


def calculate_loss(targets:torch.Tensor, logits:torch.Tensor):
    batch_size, context_length, vocab_size = logits.shape
    probs = logits.view(batch_size * context_length, vocab_size)
    ids = targets.view(batch_size * context_length)
    loss = nn.functional.cross_entropy(
        probs,
        ids
    )
    return loss

if __name__=="__main__":
    logits=torch.tensor([[[10,0,0],[0,10,0]]],dtype=torch.float32)
    targets=torch.tensor([[0,1]],dtype=torch.long)
    print("No loss:", millify(calculate_loss(targets,logits).tolist()))
    targets = torch.tensor([[1, 0]], dtype=torch.long)
    print("Big loss:", millify(calculate_loss(targets, logits).tolist()))