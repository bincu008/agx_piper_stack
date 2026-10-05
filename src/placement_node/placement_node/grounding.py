"""Pure request construction for RoboBrain grounding."""


def grounding_request(model, image_url, target):
    """Build the selected request without changing the original image."""
    prompt = (
        "Please provide the bounding box coordinate of the region this sentence describes: "
        f"{target}. Return only [x1, y1, x2, y2] in normalized 0-1000 coordinates. "
        "Cover the full visible extent of the object, including its handle if present.")
    return {
        "model": model, "temperature": 0, "seed": 1234, "max_tokens": 48,
        "ignore_eos": False, "cache_prompt": False, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt}]}],
    }
