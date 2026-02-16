import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_context_length, transformer_blocks: int, dropout: float, bias: bool,
                 residual_scaling: bool):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # IMPROVE: batch
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        # Initialize weight with residual scaling applied
        if residual_scaling:
            self.proj.weight.data *= (1 / math.sqrt(2 * transformer_blocks))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_context_length, max_context_length)).bool()
        )

    def forward(self, x):
        batch_size, context_length, embedding_dim = x.shape

        # Project once
        key = self.key(x)  # (B, T, C)
        query = self.query(x)
        value = self.value(x)

        # Split into heads
        key = key.view(batch_size, context_length, self.n_heads, self.d_head).transpose(1, 2)
        query = query.view(batch_size, context_length, self.n_heads, self.d_head).transpose(1, 2)
        value = value.view(batch_size, context_length, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        attention = (query @ key.transpose(-2, -1)) / (self.d_head ** 0.5)

        attention = attention.masked_fill(~self.mask[:context_length, :context_length], float('-inf'))
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


class ChatModel(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            embedding_dropout: float,
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

        # replace sequential, fix weight tying
        self.layers = nn.Sequential(
            Embedding(vocabulary_size, embedding_size, max_context_length, device),
            nn.Dropout(embedding_dropout),
            nn.Sequential(*[
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
            ]),
            nn.LayerNorm(embedding_size, bias=bias),
            nn.Linear(embedding_size, vocabulary_size, bias=False)
        )

        print("embedding", self.layers[0].token_emb)
        print("head", self.layers[4])
        # Weight tying
        if weight_tying:
            self.layers[4].weight = self.layers[0].token_emb.weight

    def forward(self, x, targets=None):
        logits = self.layers(x)

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
        embedding_dropout=0.0,
        max_context_length=1,
        ff_size_multiplier=1,
        transformer_blocks=2,
        attention_heads=2,
        dropout=0,
        bias=True,
    )
    for name, param in model.named_parameters():
        print(name)
