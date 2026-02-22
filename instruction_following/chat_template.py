from special_tokens import special_tokens


def chat_template(messages: list[dict[str, str]], add_generation_token: bool = False) -> str:
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
    if add_generation_token:
        result += special_tokens["assistant"]
    return result
