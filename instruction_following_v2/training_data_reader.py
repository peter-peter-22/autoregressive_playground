import random

from datasets import Dataset
import torch


def infinite_iterator(ds: Dataset, seed: int = 0):
    n = seed
    while True:
        iterator = iter(ds.shuffle(n))
        n += 1
        for el in iterator:
            yield el


class TrainingDataReader:
    def __init__(
            self,
            context_length: int,
            padding_token_id: int,
            batch_size: int,
            train_dataset: Dataset,
            test_dataset: Dataset,
            device: str,
            ignore_index: int
    ):
        self.context_length = context_length
        self.padding_token_id = padding_token_id
        self.batch_size = batch_size
        self.train_iterator = infinite_iterator(train_dataset,seed=random.randint(0,1000))
        self.test_iterator = infinite_iterator(test_dataset,seed=random.randint(0,1000))
        self.device = device
        self.ignore_index = ignore_index

    def prepare_chat(self, chat, target_size):
        """Fixate the length with padding and truncation"""
        token_ids = chat["tokens"]
        assistant_mask = chat["assistant_mask"]
        length = len(token_ids)
        # truncate to target size
        if length > target_size:
            trim = length - target_size
            return token_ids[trim:], assistant_mask[trim:]
        # pad to target size
        if length < target_size:
            padding = target_size - length
            return token_ids + [self.padding_token_id] * padding, assistant_mask + [False] * padding
        # unchanged
        return token_ids, assistant_mask

    def apply_mask(self, tokens, assistant_mask):
        return [
            t if assistant_mask[i] else self.ignore_index
            for i, t in enumerate(tokens)
        ]

    def get_batch(self,test_split:bool=True):
        batch = [next(self.test_iterator if test_split else self.train_iterator) for _ in range(self.batch_size)]
        longest_chat = max([len(row["tokens"]) for row in batch])
        target_length = min(longest_chat, self.context_length + 1)
        chats = [self.prepare_chat(chat, target_length) for chat in batch]
        x = torch.stack([torch.tensor(tokens[0:target_length - 1], dtype=torch.long) for tokens, mask in chats])
        y = torch.stack(
            [torch.tensor(self.apply_mask(tokens[1:target_length], mask[1:target_length]), dtype=torch.long) for tokens, mask
             in
             chats])
        if self.device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
