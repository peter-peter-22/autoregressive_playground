from instruction_following.tokenizer.special_tokens import special_tokens


def chat_template(messages: list[dict[str, str]]):
    result = ""
    for message in messages:
        role = message["role"]
        role_token = None
        match role:
            case "user":
                role_token = special_tokens["user"]
            case "system":
                role_token = special_tokens["system"]
            case "assistant":
                role_token = special_tokens["assistant"]
            case _:
                raise ValueError(f"Invalid message role {role}")
        result += role_token + message["content"] + special_tokens["end_of_turn"] + "\n"
    return result
