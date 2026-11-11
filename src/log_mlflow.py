
"""Enregistrer les expériences RAG existantes dans MLflow."""

import json
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments"


def main():
    # Enregistrement local, indépendant du terminal utilisé
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Qanoun_ma_RAG_Chunking")

    for exp_id in ("E1", "E2"):
        report_path = EXPERIMENTS_DIR / exp_id / "report.json"

        if not report_path.is_file():
            raise FileNotFoundError(f"Rapport introuvable : {report_path}")

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        params = report["experiment"]
        metrics = report["metrics"]

        if params["corpus_version"] != "qanoun-v1":
            raise ValueError(f"{exp_id} utilise un ancien corpus.")

        if params["id"] != exp_id:
            raise ValueError(f"Identifiant incorrect : {exp_id}")

        with mlflow.start_run(run_name=f"{exp_id}_chunking"):
            mlflow.log_params({
                "experiment_id": exp_id,
                "corpus_version": params["corpus_version"],
                "model": params["model"],
                "model_revision": params["revision"],
                "chunk_tokens": params["chunk_tokens"],
                "overlap": params["overlap"],
                "k": params["k"],
                "chunk_count": params["chunk_count"],
                "evaluation_level": params["evaluation_level"],
                "questions_sha256": params["questions_sha256"],
            })

            mlflow.log_metrics({
                "precision_at_4": metrics["precision_at_k"],
                "recall_at_4": metrics["recall_at_k"],
                "mrr": metrics["mrr"],
            })

            mlflow.log_artifact(str(report_path))

        print(f"{exp_id} enregistré dans MLflow.")

    print("Enregistrement MLflow terminé.")


if __name__ == "__main__":
    main()
