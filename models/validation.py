import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import dspy

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(text1, text2):
    """Computes cosine similarity between two text embeddings."""
    emb1 = embedding_model.encode([text1])
    emb2 = embedding_model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]


def validate_cause_effect(example, pred, threshold=0.8):
    """Validation function using cosine similarity."""
    cause_sim = compute_similarity(example.cause, pred.cause)
    effect_sim = compute_similarity(example.effect, pred.effect)
    return cause_sim >= threshold and effect_sim >= threshold


def evaluate_model(
    few_shot_examples, extractor, save_path="./data/output/evaluation_results.csv"
):
    """Evaluates model performance and saves results."""
    results = []
    correct = 0
    total = len(few_shot_examples)

    for ex in few_shot_examples:
        # Convert dictionary to DSPy Example object
        example = dspy.Example(text=ex["text"], cause=ex["cause"], effect=ex["effect"])

        pred = extractor.extract_cause_effect(example.text)  # No more AttributeError!

        if pred:
            pred_dict = {"cause": pred[0], "effect": pred[1]}
            expected_dict = {"cause": example.cause, "effect": example.effect}

            # Compute similarity scores
            cause_sim = float(compute_similarity(example.cause, pred_dict["cause"]))
            effect_sim = float(compute_similarity(example.effect, pred_dict["effect"]))
            is_correct = cause_sim >= 0.8 and effect_sim >= 0.8

            results.append(
                {
                    "text": example.text,
                    "expected_cause": example.cause,
                    "expected_effect": example.effect,
                    "predicted_cause": pred_dict["cause"],
                    "predicted_effect": pred_dict["effect"],
                    "cause_similarity": round(cause_sim, 2),
                    "effect_similarity": round(effect_sim, 2),
                    "is_correct": is_correct,
                }
            )

            if is_correct:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)

    # Save JSON format
    with open(save_path.replace(".csv", ".json"), "w") as json_file:
        json.dump(results, json_file, indent=4)

    return accuracy, df_results
