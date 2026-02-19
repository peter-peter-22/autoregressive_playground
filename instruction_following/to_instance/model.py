import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            embedding,
            n_heads,
            max_context_length,
            transformer_blocks: int,
            dropout: float,
            bias: bool,
            residual_scaling: bool
    ):
        super().__init__()

        assert embedding % n_heads == 0

        self.n_heads = n_heads
        self.embedding = embedding

        # Keep K,Q,V in the same layer
        self.kqv = nn.Linear(embedding, 3 * embedding, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout_value = dropout
        self.proj = nn.Linear(embedding, embedding, bias=bias)

        # Initialize weight with residual scaling applied
        if residual_scaling:
            self.proj.weight.data *= (1 / math.sqrt(2 * transformer_blocks))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            print("using flash attention")
        else:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.tril(torch.ones(max_context_length, max_context_length))
                           .view(1, 1, max_context_length, max_context_length))
            )

    def forward(self, x):
        batch_size, context_length, embedding_dim = x.size()

        # Split the shared layer into Q,K,V
        key, query, value = self.kqv(x).split(self.embedding, dim=2)

        # Split into heads
        key = key.view(batch_size, context_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)
        query = query.view(batch_size, context_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)
        value = value.view(batch_size, context_length, self.n_heads, embedding_dim // self.n_heads).transpose(1, 2)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout_value,
                is_causal=True
            )
        else:
            # Attention scores
            attention = (query @ key.transpose(-2, -1)) / (1.0 / math.sqrt(key.size(-1)))
            attention = attention.masked_fill(self.mask[:, :, :context_length, :context_length] == 0, float('-inf'))
            attention = nn.functional.softmax(attention, dim=-1)

            # Attention score dropout
            attention = self.dropout(attention)

            # Weighted sum
            out = attention @ value

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, context_length, embedding_dim)

        return self.proj(out)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_context_length, device: str):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_context_length, d_model)
        self.device = device

    def forward(self, x):
        batch_size, context_length = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(context_length, device=self.device))
        return tok + pos


class FeedForward(nn.Module):
    def __init__(self, embedding_size, size_multiplier, transformer_blocks: int, bias: bool, residual_scaling: bool):
        super().__init__()

        assert size_multiplier >= 1

        nn.Linear(embedding_size, size_multiplier * embedding_size, bias=bias),
        nn.GELU(),
        nn.Linear(size_multiplier * embedding_size, embedding_size, bias=bias)

        self.net = nn.Sequential(
            nn.Linear(embedding_size, size_multiplier * embedding_size),
            nn.GELU(),
            nn.Linear(size_multiplier * embedding_size, embedding_size)
        )

        # Initialize weight with residual scaling applied
        if residual_scaling:
            self.net[2].weight.data *= (1 / math.sqrt(2 * transformer_blocks))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            ff_size_multiplier: int,
            attention_heads: int,
            max_context_length: int,
            transformer_blocks: int,
            dropout: float,
            bias: bool,
            residual_scaling: bool
    ):
        super().__init__()

        # attention
        self.attention = nn.Sequential(
            nn.LayerNorm(embedding_size, bias=bias),
            MultiHeadSelfAttention(
                embedding_size,
                attention_heads,
                max_context_length,
                transformer_blocks,
                dropout,
                bias,
                residual_scaling
            ),
            nn.Dropout(dropout)
        )

        # feed forward
        self.ff = nn.Sequential(
            nn.LayerNorm(embedding_size, bias=bias),
            FeedForward(embedding_size, ff_size_multiplier, transformer_blocks, bias, residual_scaling),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ff(x)
        return x


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ChatModel(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            max_context_length: int,
            ff_size_multiplier: int,
            transformer_blocks: int,
            attention_heads: int,
            dropout: float,
            bias: bool,
            weight_tying: bool = True,
            residual_scaling: bool = True,
            device: str = "cpu"
    ):
        super().__init__()

        self.emb = Embedding(vocabulary_size, embedding_size, max_context_length, device)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                embedding_size,
                ff_size_multiplier,
                attention_heads,
                max_context_length,
                transformer_blocks,
                dropout,
                bias,
                residual_scaling
            )
            for _ in range(transformer_blocks)
        ])
        self.ln = nn.LayerNorm(embedding_size, bias=bias)
        self.head = nn.Linear(embedding_size, vocabulary_size, bias=False)

        # Weight tying
        # The order does not count, but the weight initialization does
        # head=emb: 130 loss <1, 220 loss <0.1,
        # emb=head: 130 loss <1, 230 loss <0.1
        # Without custom (3x smaller than Kaiming uniform (default)) weight initialization:
        # Either 130 starting loss or the loss doesn't go below 5 while training.
        # Since the custom weight initialization is the same for the embedding and projection layers, the order no longer counts
        if weight_tying:
            self.emb.token_emb.weight = self.head.weight

        self.apply(_init_weights)

    def forward(self, x, targets=None):
        x = self.emb(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits

        batch_size, context_length, vocab_size = logits.shape
        probs = logits.view(batch_size * context_length, vocab_size)  # probabilities: B * T, V
        ids = targets.view(batch_size * context_length)  # ids: B * T
        loss = nn.functional.cross_entropy(
            probs,
            ids
        )
        return logits, loss


if __name__ == "__main__":
    model = ChatModel(
        vocabulary_size=10,
        embedding_size=2,
        max_context_length=1,
        ff_size_multiplier=1,
        transformer_blocks=2,
        attention_heads=2,
        dropout=0,
        bias=True,
    )
    for name, param in model.named_parameters():
        print(name)
