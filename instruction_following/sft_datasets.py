from datasets import load_from_disk, Dataset, interleave_datasets


def get_sft_ds(names: list[str], prefix: str = "tokenized_data/") -> Dataset:
    dataset_list = [load_from_disk(prefix + name) for name in names]
    sizes = [ds.num_rows for ds in dataset_list]
    sizes = [size / sum(sizes) for size in sizes]
    return interleave_datasets(
        dataset_list,
        probabilities=sizes,
        stopping_strategy="all_exhausted"
    )
