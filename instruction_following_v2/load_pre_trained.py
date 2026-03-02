import os

from huggingface_hub import login
from millify import millify
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # GPT-2 120M

# Login to prevent download rate limit
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no pad token by default — add one for batching
tokenizer.pad_token = tokenizer.eos_token

# Load pretrained architecture + weights
model = AutoModelForCausalLM.from_pretrained(model_name)

if __name__ == "__main__":
    print(model)
    print(model.config)
    print("EOS id:", model.config.eos_token_id)
    print("Special tokens:", tokenizer.special_tokens_map)
    print("Context length:", model.config.n_positions)
    print("Total parameters:", millify(sum(p.numel() for p in model.parameters())))
