from load_pre_trained import tokenizer, model

eos_id = model.config.eos_token_id


def sanitize_user_text(text):
    return text.replace("<|endoftext|>", "<|end of text|>")


def encode_chat(instruction: str, output: str | None = None, input: str | None = None,
                escape_special_tokens: bool = True):
    # Sanitize user text
    if escape_special_tokens:
        instruction = sanitize_user_text(instruction)
        if input is not None:
            input = sanitize_user_text(input)

    # Build text
    text = f"### Instruction:\n{instruction}"
    if input is not None:
        text += f"\n\n### Input:\n{input}"
    text += f"\n\n### Response:\n"  # The generation token is added even if there is no output
    if output is not None:
        text += output

    # Encode
    token_ids = tokenizer.encode(text, )
    if output is not None:  # If the output is defined, add the EOS token
        token_ids.append(eos_id)

    return token_ids
