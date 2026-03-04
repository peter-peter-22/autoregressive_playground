import torch
import torch.nn as nn
from tokenizers import Tokenizer
from chat_template import encode_chat
from training_loop import SimpleInference


def apply_top_p(logits: torch.Tensor, p: float, min_keep: int = 1) -> torch.Tensor:
    """
    In-place top-p filtering on logits
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_idx = logits.sort(descending=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    cum_probs = sorted_probs.cumsum(dim=-1)

    # Mask tokens after the top-p cutoff
    mask = cum_probs > p

    # Protect at least min_keep tokens
    mask[..., :min_keep] = False

    # Scatter back
    to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_idx, src=mask
    )

    logits[to_remove] = -float('inf')
    return logits

class ChatCompletion:
    def __init__(
            self,
            tokenizer: Tokenizer,
            inference: SimpleInference,
            device:str,
            stop_token_ids:list[int],
            max_context_length:int,
            top_p: float = 0.9,
            top_k: int | None = None,
            temperature: float = 0.7,
            max_new_tokens: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.stop_token_ids = stop_token_ids
        self.device = device
        self.max_context_length = max_context_length
        self.max_new_tokens=max_new_tokens
        self.inference = inference


    @torch.no_grad()
    def generate(self, instruction:str,input:str|None=None):

        # Encode chat
        idx = torch.tensor(
            [encode_chat(instruction=instruction, input=input)],
            device=self.device,
            dtype=torch.long,
        )

        for _ in range(self.max_new_tokens):
            # Get logits
            idx_cond = idx[:, -self.max_context_length:]
            logits = self.inference(idx_cond)
            logits = logits[:, -1, :]

            # Top k
            if self.top_k is not None:
                top_logits, top_pos = torch.topk(logits, self.top_k)
                logits = torch.where(
                    logits < top_logits[:, -1],
                    input=torch.tensor(float("-inf")),
                    other=logits
                )

            # Top p
            logits=apply_top_p(logits,self.top_p)

            # Sampling with temperature
            probs = nn.functional.softmax(logits / self.temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            # Handle stop tokens if needed
            if self.stop_token_ids and next_id.item() in self.stop_token_ids:
                print("Reached stop token")
                break

        # Decode
        return self.tokenizer.decode(idx.tolist())