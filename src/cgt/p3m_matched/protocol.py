"""The matched ShotBench protocol (docs/p3m_matched_protocol.md section 1.2-1.3 and 2): prompt, system prompts, decoding.
Replicates the official ShotBench inference script (Vchitect/ShotBench evaluation/shotvl/evaluate.py). Pure Python: no model, no data."""
import hashlib
import json

PROTOCOL_VERSION = "p3m-matched-v1"
USER_TAIL = "Please select the most likely answer from the options above."  # no trailing space/newline (unlike ShotQA sft turns)
SYSTEM_PLAIN = "You are a helpful assistant."
SYSTEM_REASONING = (  # the official --reasoning system prompt: five literal segments, single spaces, ends with "tags."
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. The reasoning process and answer are enclosed within <think> "
    "</think> and <answer> </answer> tags."
)
SYSTEM_PROMPTS = {"plain": SYSTEM_PLAIN, "reasoning": SYSTEM_REASONING}
MAX_NEW_TOKENS = 1024
DECODING = {"do_sample": False, "temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS, "skip_special_tokens": True}
PARSER_VERSION = "shotbench-calculate_scores-local-v1 (answer tag, option tokens, option text; no GPT fallback; unmatched = invalid = wrong)"


def build_user_text(question: str, options: dict[str, str]) -> str:
    """`Question: {q}\\nOptions:\\n{k}. {v}...\\nPlease select the most likely answer from the options above.` with the options in their
    STORED order (no sorting) and the question unchanged, exactly as the official script builds it."""
    opts_block = "Options:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question: {question}\n{opts_block}\n" + USER_TAIL


def build_messages(question: str, options: dict[str, str], variant: str) -> list[dict]:
    """[system, user(image first, then text)] in the HF chat-template content format."""
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown system-prompt variant {variant!r}; allowed {sorted(SYSTEM_PROMPTS)}")
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[variant]}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_user_text(question, options)}]}]


def protocol_spec(variant: str) -> dict:
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown system-prompt variant {variant!r}")
    return {"version": PROTOCOL_VERSION, "variant": variant, "system_prompt": SYSTEM_PROMPTS[variant], "user_tail": USER_TAIL,
            "user_template": "Question: {question}\\nOptions:\\n{k}. {v} (stored order, joined by \\n)\\n" + USER_TAIL,
            "message_order": "system, user[image, text]", "decoding": DECODING, "parser": PARSER_VERSION}


def protocol_sha256(variant: str) -> str:
    return hashlib.sha256(json.dumps(protocol_spec(variant), sort_keys=True).encode()).hexdigest()


def question_from_shotqa_turn(user_text: str) -> str:
    """The question of a ShotQA sft user turn (`Question: <q>\\nOptions:...`), unchanged."""
    first = user_text.split("\n", 1)[0]
    if not first.startswith("Question: "):
        raise ValueError("not a ShotQA question turn")
    return first.removeprefix("Question: ")
