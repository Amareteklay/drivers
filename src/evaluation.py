# src/evaluation.py
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from dspy.evaluate.evaluate import Evaluate

# Load a SentenceTransformer model once for semantic evaluation.
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

def cause_effect_exact_match(example, prediction):
    gold_cause = example.cause.strip().lower()
    gold_effect = example.effect.strip().lower()
    pred_cause = prediction.cause.strip().lower()
    pred_effect = prediction.effect.strip().lower()
    score_cause = float(gold_cause == pred_cause)
    score_effect = float(gold_effect == pred_effect)
    return 0.5 * (score_cause + score_effect)

def cause_effect_semantic_match(example, prediction):
    gold_cause = example.cause.strip()
    gold_effect = example.effect.strip()
    pred_cause = prediction.cause.strip()
    pred_effect = prediction.effect.strip()
    
    gold_cause_emb = model_sbert.encode(gold_cause, convert_to_tensor=True)
    gold_effect_emb = model_sbert.encode(gold_effect, convert_to_tensor=True)
    pred_cause_emb = model_sbert.encode(pred_cause, convert_to_tensor=True)
    pred_effect_emb = model_sbert.encode(pred_effect, convert_to_tensor=True)
    
    cause_sim = F.cosine_similarity(gold_cause_emb, pred_cause_emb, dim=0).item()
    effect_sim = F.cosine_similarity(gold_effect_emb, pred_effect_emb, dim=0).item()
    return (cause_sim + effect_sim) / 2.0

def run_evaluation(model, devset, metric, display_table=5, display_progress=True, return_all_scores=False):
    evaluator = Evaluate(
        devset=devset,
        display_table=display_table,
        display_progress=display_progress
    )
    return evaluator(model, metric=metric, return_all_scores=return_all_scores)
