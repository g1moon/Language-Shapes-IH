"""
Evaluation function for Korean translation quality using chrF++ metric.
chrF++ is character n-gram F-score with word n-grams, well-suited for Korean.
"""

import sacrebleu

# Initialize chrF++ metric ONCE and reuse
# word_order=2 enables chrF++ (chrF with word bigrams)
_CHRF_METRIC = sacrebleu.CHRF(word_order=2)


def eval_chrf(answer: str, prediction: str, loose: bool = False) -> float:
    """
    Evaluate Korean translation quality using chrF++.

    Args:
        answer: Reference Korean translation (ground truth)
        prediction: Model's Korean translation output
        loose: If True, try various text cleaning strategies and return max score

    Returns:
        chrF++ score normalized to 0-1 scale
    """
    if loose:
        # Try various text cleaning strategies for loose evaluation
        p = prediction.split("\n")
        prediction_remove_first = "\n".join(p[1:]).strip()
        prediction_remove_last = "\n".join(p[:-1]).strip()
        prediction_remove_both = "\n".join(p[1:-1]).strip()
        revised_prediction = prediction.replace("*", "").strip()
        revised_prediction_remove_first = prediction_remove_first.replace("*", "").strip()
        revised_prediction_remove_last = prediction_remove_last.replace("*", "").strip()
        revised_prediction_remove_both = prediction_remove_both.replace("*", "").strip()

        all_predictions = [
            prediction.strip(),
            revised_prediction,
            prediction_remove_first,
            prediction_remove_last,
            prediction_remove_both,
            revised_prediction_remove_first,
            revised_prediction_remove_last,
            revised_prediction_remove_both,
        ]

        all_scores = []
        for p in all_predictions:
            if p:
                all_scores.append(chrf_score(answer, p))
            else:
                all_scores.append(0.0)

        return max(all_scores)
    else:
        return chrf_score(answer, prediction)


def chrf_score(answer: str, prediction: str) -> float:
    """
    Calculate chrF++ score.

    Args:
        answer: Reference Korean text
        prediction: Predicted Korean text

    Returns:
        chrF++ score normalized to 0-1 scale
    """
    if not prediction or not prediction.strip():
        return 0.0

    if not answer or not answer.strip():
        return 0.0

    result = _CHRF_METRIC.sentence_score(
        hypothesis=prediction.strip(),
        references=[answer.strip()],
    )

    # Normalize score from 0-100 to 0-1
    return result.score / 100.0


def chrf_recall_score(answer: str, prediction: str) -> float:
    """
    Calculate chrF++ recall score.
    Measures how much of the reference's character/word n-grams appear in the hypothesis.
    This directly captures "does the output contain the answer's content?",
    which is ideal for detecting whether translation was performed.

    Args:
        answer: Reference text (ground truth)
        prediction: Model's output text

    Returns:
        chrF++ recall score (0-1 scale)
    """
    if not prediction or not prediction.strip():
        return 0.0
    if not answer or not answer.strip():
        return 0.0

    # Strip markdown formatting (*, #) — these break word n-gram matching
    # but don't appear in reference translations
    cleaned = prediction.replace("*", "").replace("#", "").strip()
    if not cleaned:
        return 0.0

    stats = _CHRF_METRIC._extract_corpus_statistics(
        [cleaned], [[answer.strip()]]
    )[0]

    total_recall = 0.0
    effective = 0
    for i in range(_CHRF_METRIC.order):
        _, n_ref, n_match = stats[3 * i : 3 * i + 3]
        if n_ref > 0:
            total_recall += n_match / n_ref
            effective += 1

    return total_recall / effective if effective > 0 else 0.0


def eval_chrf_recall(answer: str, prediction: str) -> float:
    """
    Evaluate translation detection using chrF++ recall.

    Markdown formatting (*, #) is stripped inside chrf_recall_score().
    Line-removal variants are unnecessary for recall: removing hypothesis text
    can only decrease or maintain matched n-grams, never increase them.

    Args:
        answer: Reference text (ground truth)
        prediction: Model's output text

    Returns:
        chrF++ recall score (0-1 scale)
    """
    return chrf_recall_score(answer, prediction)


def corpus_chrf_score(answers: list, predictions: list) -> float:
    """
    Calculate corpus-level chrF++ score.

    Args:
        answers: List of reference Korean texts
        predictions: List of predicted Korean texts

    Returns:
        Corpus chrF++ score normalized to 0-1 scale
    """
    if not predictions or not answers:
        return 0.0

    result = _CHRF_METRIC.corpus_score(
        hypotheses=[p.strip() for p in predictions],
        references=[[a.strip() for a in answers]],
    )

    return result.score / 100.0
