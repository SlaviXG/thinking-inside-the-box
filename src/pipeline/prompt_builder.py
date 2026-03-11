_SYSTEM_PROMPT = (
    "You are an expert financial forensics AI. Review the provided transaction logs. "
    "Look for signs of money laundering, such as rapid structuring, circular flow, "
    "or pass-through accounts. "
    "Provide a short, step-by-step reasoning process, then a final conclusion."
)


def build_investigation_prompt(account_id: str, graph_context: str) -> list[dict]:
    """
    Pure function. Returns a messages list in chat format for apply_chat_template.
    The exact prompt wording is preserved from the working notebook prototype.
    """
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Investigate the following transactions for suspicious activity:"
                f"\n\n{graph_context}"
            ),
        },
    ]
