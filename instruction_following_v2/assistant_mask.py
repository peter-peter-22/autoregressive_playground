from load_pre_trained import tokenizer, model

output_token_ids = tokenizer.encode("### Response:\n")
response_len = len(output_token_ids)
eos_id = model.config.eos_token_id


def assistant_mask(token_ids: list[int]):
    message_len = len(token_ids)
    mask = [False] * message_len
    found_output_tokens = 0
    for i in range(message_len):
        token_id = token_ids[i]
        if found_output_tokens == response_len:
            mask[i] = True
            if token_id == eos_id:
                break
            continue
        if token_id == output_token_ids[found_output_tokens]:
            found_output_tokens += 1
    return mask
