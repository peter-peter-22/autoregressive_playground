import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_context_length):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model)

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
    def __init__(self, embedding_size, size_multiplier):
        super().__init__()

        assert size_multiplier >= 1

        self.net = nn.Sequential(
            nn.Linear(embedding_size, size_multiplier * embedding_size),
            nn.GELU(),
            nn.Linear(size_multiplier * embedding_size, embedding_size)
        )

    def forward(self, x):
        return x + self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            ff_size_multiplier: int,
            attention_heads: int,
            max_context_length: int,
            attention_dropout: float,
            ff_dropout: float
    ):
        super().__init__()

        # attention
        self.attention = nn.Sequential(
            nn.LayerNorm(embedding_size),
            MultiHeadSelfAttention(embedding_size, attention_heads, max_context_length),
            nn.Dropout(attention_dropout)
        )

        # feed forward
        self.ff = nn.Sequential(
            nn.LayerNorm(embedding_size),
            FeedForward(embedding_size, ff_size_multiplier),
            nn.Dropout(ff_dropout)
        )

    def forward(self, x):
        x += self.attention(x)
        x += self.ff(x)
        return x


class ChatModel(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_size: int,
            embedding_dropout: float,
            attention_dropout: float,
            max_context_length: int,
            ff_size_multiplier: int,
            ff_dropout: float,
            transformer_blocks: int,
            attention_heads: int,
            device: str = "cpu"
    ):
        super().__init__()

        self.layers = nn.Sequential(
            Embedding(vocabulary_size, embedding_size, max_context_length, device),
            nn.Dropout(embedding_dropout),
            nn.Sequential(*[
                TransformerBlock(
                    embedding_size,
                    ff_size_multiplier,
                    attention_heads,
                    max_context_length,
                    attention_dropout,
                    ff_dropout
                )
                for _ in range(transformer_blocks)
            ]),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, vocabulary_size)
        )

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
