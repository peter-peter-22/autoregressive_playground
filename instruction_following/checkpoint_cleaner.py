import os


class CheckpointCleaner:
    def __init__(self, max_checkpoints: int, preserve_checkpoints: list | None = None):
        self.max_checkpoints = max_checkpoints
        self.history: list[str] = []
        self.preserve_checkpoints = preserve_checkpoints

    def step(self, new_checkpoint_path: str):
        self.history.append(new_checkpoint_path)
        count = len(self.history)
        if count > self.max_checkpoints:
            remove = count - self.max_checkpoints
            for i in range(remove):
                path = self.history[i]
                if not self.preserve_checkpoints or path not in self.preserve_checkpoints:
                    try:
                        os.remove(path)
                        print(f"Removed checkpoint {path}")
                    except FileNotFoundError:
                        print(f"The removed checkpoint {path} does not exist")
                else:
                    print(f"Preserving checkpoint {path}")
            self.history = self.history[remove:]


if __name__ == "__main__":
    checkpoint_cleaner = CheckpointCleaner(max_checkpoints=5)
    for i in range(10):
        checkpoint_cleaner.step(f"checkpoints/checkpoint_{i}")
