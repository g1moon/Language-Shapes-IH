"""
Persona evaluation functions.
Extracts verdicts from LLM judge outputs and scores them against ground truth labels.
"""

import re


def extract_verdict(judge_output: str):
    """
    Extract verdict (A, B, or C) from judge output text.

    Tries patterns in order:
      1. [[A]], [[B]], [[C]] (standard format)
      2. [A], [B], [C] (single bracket)
      3. **[[A]]**, **[[B]]**, **[[C]]** (markdown bold)

    Returns the last match found, or None if no match.
    """
    if not judge_output:
        return None

    patterns = [
        r'\[\[([ABC])\]\]',           # [[A]], [[B]], [[C]]
        r'\[([ABC])\]',               # [A], [B], [C]
        r'\*\*\[\[([ABC])\]\]\*\*',   # **[[A]]**, **[[B]]**, **[[C]]**
    ]

    verdict = None
    for pattern in patterns:
        matches = re.findall(pattern, judge_output)
        if matches:
            verdict = matches[-1]  # last match

    return verdict


def eval_persona(verdict: str, label: int) -> int:
    """
    Map verdict to score given ground truth label.

    Args:
        verdict: "A", "B", or "C"
        label: 0 (correct persona is A) or 1 (correct persona is B)

    Returns:
        1 if correct, 0 otherwise
    """
    if verdict == "C":
        return 0

    correct_verdict = "A" if label == 0 else "B"
    return 1 if verdict == correct_verdict else 0
