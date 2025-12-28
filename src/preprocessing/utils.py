def chat_structure(element):
    """
    Takes a single element from a dataset and returns a dictionary with the chat structure.
    :param element:
    :return:
    """
    assert "prompt" in element, "Prompt must be present in the element."
    element["prompt"] = [{"role": "user", "content": element["prompt"]}]

    if "chosen" in element:
        for name in ["chosen", "rejected"]:
            element[name] = element["prompt"] + [
                {"role": "assistant", "content": element[name]}
            ]
    if "completion" in element:
        element["completion"] = [
            {"role": "assistant", "content": element["completion"]}
        ]
    return element


def get_chat_prompt(
    example: dict, system_prompt: str = None, follower_prompt: str = None
) -> dict:
    if system_prompt:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
    else:
        prompt = {"prompt": [{"role": "user", "content": example["prompt"]}]}

    if follower_prompt:
        prompt += [
            {"role": "assistant", "content": example["completion"]},
            {"role": "user", "content": follower_prompt},
        ]
    return {"prompt": prompt}
