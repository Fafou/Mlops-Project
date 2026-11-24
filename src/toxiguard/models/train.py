"""Entraînement du modèle de base avec suivi des expériences dans MLflow.

Usage : poetry run python -m toxiguard.models.train --max-features 50000 --c 4.0
"""

import argparse

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from toxiguard.config import LABELS, MODELS_DIR, RANDOM_STATE
from toxiguard.data.load import load_jigsaw
from toxiguard.models.pipeline import build_pipeline


def main(max_features: int, c: float, threshold: float = 0.5) -> dict:
    df = load_jigsaw()
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df[LABELS], test_size=0.2, random_state=RANDOM_STATE
    )

    mlflow.set_experiment("toxiguard-baseline")
    with mlflow.start_run():
        params = {"model": "tfidf+logreg", "max_features": max_features, "C": c, "threshold": threshold}
        mlflow.log_params(params)

        model = build_pipeline(max_features=max_features, c=c)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)
        pred = (proba >= threshold).astype(int)
        metrics = {
            "roc_auc_macro": roc_auc_score(y_test, proba, average="macro"),
            "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        }
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / "baseline.joblib")
        mlflow.sklearn.log_model(model, "model")

    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le modèle de base ToxiGuard.")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args.max_features, args.c, args.threshold)
