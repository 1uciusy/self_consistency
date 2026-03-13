from typing import List, Optional
import re

def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from the reasoning path."""
    # Look for GSM8K format: #### [answer]
    gsm8k_match = re.search(r"####\s*([\d.]+)", text)
    if gsm8k_match:
        return gsm8k_match.group(1)

    # Look for the last number in the text as fallback
    numbers = re.findall(r"([\d.]+)", text)
    return numbers[-1] if numbers else None

def majority_vote(answers: List[str]) -> Optional[str]:
    # important for consistency
    valid_answers = [ans for ans in answers if ans is not None]
    if not valid_answers:
        return None
    # Convert answers to float for comparison
    float_answers = []
    for ans in valid_answers:
        try:
            float_answers.append(float(ans))
        except ValueError:
            continue

    grouped_answers = []
    for ans in float_answers:
        found_group = False
        for group in grouped_answers:
            if abs(group[0] - ans) < 0.1:
                group.append(ans)
                found_group = True
                break
        if not found_group:
            grouped_answers.append([ans])
    
    largest_group = max(grouped_answers, key=len)
    modal_answer = sum(largest_group) / len(largest_group)
    
    return str(int(modal_answer)) if modal_answer.is_integer() else str(modal_answer)

def check_answer_correctness(predicted: Optional[str], correct: str, tolerance: float = 0.1) -> bool:

    if predicted is None:
        return False
        
    try:
        pred_float = float(predicted)
        correct_float = float(correct)
        return abs(pred_float - correct_float) < tolerance
    except (ValueError, TypeError):
        return False