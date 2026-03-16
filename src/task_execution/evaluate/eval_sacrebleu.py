"""
Evaluation function for Korean translation quality using SacreBLEU with flores200 tokenizer.
"""

import sacrebleu

# Initialize BLEU metric ONCE and reuse to prevent memory leak
# (sacrebleu.sentence_bleu() creates a new tokenizer object every call,
#  causing severe memory leak with flores200 SentencePiece tokenizer)
_BLEU_METRIC = sacrebleu.BLEU(tokenize='flores200', effective_order=True)


def eval_sacrebleu(answer: str, prediction: str, loose: bool = False) -> float:
    """
    Evaluate Korean translation quality using SacreBLEU with flores200 tokenizer.

    Args:
        answer: Reference Korean translation (ground truth)
        prediction: Model's Korean translation output
        loose: If True, try various text cleaning strategies and return max score

    Returns:
        BLEU score (0-100 scale normalized to 0-1)
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
            if p:  # Only evaluate non-empty predictions
                all_scores.append(sacrebleu_score(answer, p))
            else:
                all_scores.append(0.0)
        
        return max(all_scores)
    else:
        return sacrebleu_score(answer, prediction)


def sacrebleu_score(answer: str, prediction: str) -> float:
    """
    Calculate SacreBLEU score with flores200 tokenizer.

    Args:
        answer: Reference Korean text
        prediction: Predicted Korean text

    Returns:
        BLEU score normalized to 0-1 scale
    """
    if not prediction or not prediction.strip():
        return 0.0

    if not answer or not answer.strip():
        return 0.0

    # Reuse the global BLEU metric object (tokenizer loaded only once)
    bleu = _BLEU_METRIC.sentence_score(
        hypothesis=prediction.strip(),
        references=[answer.strip()],
    )

    # Normalize score from 0-100 to 0-1
    return bleu.score / 100.0


def corpus_sacrebleu_score(answers: list, predictions: list) -> float:
    """
    Calculate corpus-level SacreBLEU score with flores200 tokenizer.

    Args:
        answers: List of reference Korean texts
        predictions: List of predicted Korean texts

    Returns:
        Corpus BLEU score normalized to 0-1 scale
    """
    if not predictions or not answers:
        return 0.0

    # Reuse the global BLEU metric object
    bleu = _BLEU_METRIC.corpus_score(
        hypotheses=[p.strip() for p in predictions],
        references=[[a.strip() for a in answers]],
    )

    return bleu.score / 100.0
