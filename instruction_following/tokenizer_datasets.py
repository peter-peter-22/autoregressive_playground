import os
from typing import Iterator

from datasets import load_dataset
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

login(hf_token)


def tokenizer_lite_dataset():
    total_length = 0

    # dataset
    no_robots_ds = load_dataset("HuggingFaceH4/no_robots", split="test").select_columns(
        ["messages"])
    total_length += no_robots_ds.dataset_size

    # iterator
    def iterator() -> Iterator[str]:
        for row in no_robots_ds:
            for msg in row["messages"]:
                yield msg["content"]

    return iterator, total_length


def tokenizer_real_dataset():
    total_length = 0

    # no robots
    no_robots_ds = load_dataset("HuggingFaceH4/no_robots", split="test").select_columns(["messages"])
    total_length += no_robots_ds.dataset_size

    # wiki
    wiki_ds = load_dataset("jordiclive/wikipedia-summary-dataset", split="train").select_columns(["summary"])
    reduced_count = 500_000
    wiki_ds = wiki_ds.take(reduced_count)
    total_length += reduced_count

    # tiny stories
    tiny_stories_ds = load_dataset("roneneldan/TinyStories", split="validation").select_columns(
        ["text"])
    total_length += tiny_stories_ds.dataset_size

    # tiny textbooks
    tiny_textbooks_ds = load_dataset("nampdn-ai/tiny-textbooks", split="test").select_columns(
        ["textbook"])
    total_length += tiny_textbooks_ds.dataset_size

    # iterator
    def iterator() -> Iterator[str]:
        for row in no_robots_ds:
            for msg in row["messages"]:
                yield msg["content"]
        print("no robots completed")
        for row in wiki_ds:
            yield row["summary"]
        print("wiki completed")
        for row in tiny_stories_ds:
            yield row["text"]
        print("tiny stories completed")
        for row in tiny_textbooks_ds:
            yield row["textbook"]
        print("tiny textbooks completed")

    return iterator, total_length
