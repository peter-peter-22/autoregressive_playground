from datasets import Dataset


class TrainingDataReader:
    def __init__(self, context_length: int, padding_token_id: int, batch_size: int, dataset: Dataset, device: str):
        self.context_length = context_length
        self.padding_token_id = padding_token_id
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
