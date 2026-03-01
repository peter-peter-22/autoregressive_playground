from datasets import Dataset


class TrainingDataReader:
    def __init__(
            self,
            context_length: int,
            padding_token_id: int,
            batch_size: int,
            dataset: Dataset,
            device: str,
            ignore_index: int,
    ):
        self.context_length = context_length
        self.padding_token_id = padding_token_id
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device

    def prepare_chat(chat, target_size, pad_element):
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
            return token_ids + [pad_element] * padding, assistant_mask + [False] * padding
        # unchanged
        return token_ids, assistant_mask

    def apply_mask(tokens, assistant_mask):
        return [
            t if assistant_mask[i] else ignore_index
            for i, t in enumerate(tokens)
        ]

    def get_batch(split):
        iterator = ds_train if split == 'train' else ds_test
        batch = [next(iterator) for _ in range(batch_size)]
        longest_chat = max([len(row["tokens"]) for row in batch])
        target_length = min(longest_chat, block_size + 1)
        chats = [prepare_chat(chat, target_length, pad_id) for chat in batch]
        x = torch.stack([torch.tensor(tokens[0:target_length - 1], dtype=torch.long) for tokens, mask in chats])
        y = torch.stack(
            [torch.tensor(apply_mask(tokens[1:target_length], mask[1:target_length]), dtype=torch.long) for tokens, mask
             in
             chats])
        if device == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    # %%
