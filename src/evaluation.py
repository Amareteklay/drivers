import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from dspy.evaluate.evaluate import Evaluate
import ast

# Load SentenceTransformer for semantic evaluation.
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

def parse_extracted_pairs(pairs):
    """
    If pairs is a string representation of a list of dictionaries, convert it to a list.
    Otherwise, return as is.
    """
    if isinstance(pairs, str):
        try:
            parsed = ast.literal_eval(pairs)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except Exception as e:
            print(f"Parsing error: {e}")
            return []
    return pairs

def extracted_pairs_exact_match(example, prediction):
    """
    Computes an exact match score for the extracted cause-effect pairs.
    Converts the list of pair dictionaries into sets of tuples (cause, effect)
    after lowercasing and stripping. Returns 1.0 if they match exactly, else 0.0.
    """
    gold_pairs = parse_extracted_pairs(example.extracted_pairs)
    pred_pairs = parse_extracted_pairs(prediction.extracted_pairs)

    if not gold_pairs or not pred_pairs:
        return 0.0

    try:
        gold_set = set((pair["cause"].strip().lower(), pair["effect"].strip().lower()) for pair in gold_pairs)
        pred_set = set((pair["cause"].strip().lower(), pair["effect"].strip().lower()) for pair in pred_pairs)
    except KeyError as e:
        print(f"KeyError: {e}")
        return 0.0

    return 1.0 if gold_set == pred_set else 0.0

def extracted_pairs_semantic_match(example, prediction):
    """
    Computes a semantic similarity score for the extracted cause-effect pairs.
    For each predicted pair, finds the best matching gold pair using cosine similarity,
    then returns the average of these best scores.
    """
    gold_pairs = parse_extracted_pairs(example.extracted_pairs)
    pred_pairs = parse_extracted_pairs(prediction.extracted_pairs)

    if not gold_pairs or not pred_pairs:
        return 0.0

    sims = []
    for pred in pred_pairs:
        pred_cause = pred["cause"].strip()
        pred_effect = pred["effect"].strip()
        pred_cause_emb = model_sbert.encode(pred_cause, convert_to_tensor=True)
        pred_effect_emb = model_sbert.encode(pred_effect, convert_to_tensor=True)
        best_sim = 0.0
        for gold in gold_pairs:
            gold_cause = gold["cause"].strip()
            gold_effect = gold["effect"].strip()
            gold_cause_emb = model_sbert.encode(gold_cause, convert_to_tensor=True)
            gold_effect_emb = model_sbert.encode(gold_effect, convert_to_tensor=True)
            cause_sim = F.cosine_similarity(gold_cause_emb, pred_cause_emb, dim=0).item()
            effect_sim = F.cosine_similarity(gold_effect_emb, pred_effect_emb, dim=0).item()
            sim = (cause_sim + effect_sim) / 2.0
            best_sim = max(best_sim, sim)
        sims.append(best_sim)
    return sum(sims) / len(sims)

def run_evaluation(model, devset, metric, display_table=5, display_progress=True, return_all_scores=False):
    """
    Runs evaluation on the provided dev set using DSPy's Evaluate class and the given metric.
    """
    evaluator = Evaluate(
        devset=devset,
        display_table=display_table,
        display_progress=display_progress
    )
    return evaluator(model, metric=metric, return_all_scores=return_all_scores)
