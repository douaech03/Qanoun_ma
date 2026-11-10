
"""Réévaluer les index expérimentaux existants de MarocLex AI."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src import evaluate, ingest
from src.compare_chunking import (
    load_experiments,
    evaluate_experiment,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 1. Ouvrir un index expérimental existant
# ==================================================

def open_experiment(exp, config):
    """Ouvrir un index et vérifier sa configuration d'origine."""

    exp_dir = ROOT / "artifacts" / "experiments" / exp["id"]
    index_dir = exp_dir / "index"
    original_report_path = exp_dir / "report.json"

    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"Index introuvable : {index_dir}"
        )

    if not original_report_path.is_file():
        raise FileNotFoundError(
            f"Rapport initial introuvable : {original_report_path}"
        )

    with open(original_report_path, encoding="utf-8") as file:
        original_report = json.load(file)

    info = original_report["experiment"]

    expected = {
        "id": exp["id"],
        "chunk_tokens": exp["chunk_tokens"],
        "overlap": exp["overlap"],
        "model": config["model"],
        "revision": str(config["revision"]),
        "corpus_version": str(config["corpus_version"]),
        "collection": config["collection"],
        "max_distance": config.get("max_distance"),
    }

    for key, expected_value in expected.items():
        if info.get(key) != expected_value:
            raise ValueError(
                f"{exp['id']} : configuration différente pour "
                f"{key}. Index non réévalué."
            )

    # Vérifier que les PDF n'ont pas changé
    documents = ingest.load_manifest(
        ROOT / config["manifest"]
    )

    current_hashes = {
        doc["document_id"]: sha256_file(doc["pdf_path"])
        for doc in documents
    }

    if info.get("document_hashes") != current_hashes:
        raise ValueError(
            f"{exp['id']} : le corpus PDF a changé."
        )

    client = chromadb.PersistentClient(
        path=str(index_dir)
    )

    collection = client.get_collection(
        name=config["collection"]
    )

    actual_count = collection.count()
    expected_count = info["chunk_count"]

    if actual_count != expected_count:
        raise ValueError(
            f"{exp['id']} : {actual_count} chunks trouvés, "
            f"{expected_count} attendus."
        )

    return collection, original_report


# ==================================================
# 2. Réévaluer une expérience
# ==================================================

def reevaluate_experiment(
    exp, collection, original_report, model, questions, k, config
):
    """Calculer les nouvelles métriques sur un index existant."""

    metrics, details = evaluate_experiment(
        collection,
        model,
        questions,
        k,
        config,
    )

    question_path = (
        ROOT / "data" / "evaluation" / "questions.yaml"
    )

    questions_hash = sha256_file(question_path)

    report = {
        "experiment": {
            **original_report["experiment"],
            "reevaluation_date_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "k": k,
            "questions_count": len(questions),
            "questions_sha256": questions_hash,
        },
        "metrics": metrics,
        "details": details,
    }

    return report


# ==================================================
# 3. Enregistrer les nouveaux résultats
# ==================================================

def save_new_report(exp, report):
    """Créer un rapport sans modifier l'ancien."""

    exp_dir = (
        ROOT / "artifacts" / "experiments" / exp["id"]
    )

    count = report["experiment"]["questions_count"]
    question_hash = report["experiment"]["questions_sha256"][:12]

    filename = f"report_eval_{count}q_{question_hash}.json"

    output_path = exp_dir / filename

    # Mode x : refuse de remplacer un fichier existant
    with open(output_path, "x", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return output_path


# ==================================================
# 4. Programme principal
# ==================================================

def main():
    """Réévaluer une ou plusieurs expériences E1 à E5."""

    parser = argparse.ArgumentParser(
        description="Réévaluer les index existants"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Exemple : E1. Par défaut : toutes les expériences.",
    )

    args = parser.parse_args()

    k, experiments = load_experiments()
    config = ingest.load_config()
    questions = evaluate.load_questions()

    if args.experiment:
        experiments = [
            exp for exp in experiments
            if exp["id"] == args.experiment
        ]

        if not experiments:
            parser.error("Expérience inconnue.")

    # Vérifier les index et rapports AVANT de charger le modèle
    available = []

    questions_hash = sha256_file(
        ROOT / "data" / "evaluation" / "questions.yaml"
    )[:12]

    for exp in experiments:
        collection, original_report = open_experiment(
            exp, config
        )

        output_path = (
            ROOT / "artifacts" / "experiments" / exp["id"]
            / f"report_eval_{len(questions)}q_{questions_hash}.json"
        )

        if output_path.exists():
            raise FileExistsError(
                f"Le rapport existe déjà : {output_path}"
            )

        available.append((exp, collection, original_report))

    print(f"\nQuestions validées : {len(questions)}")
    print(f"Expériences à réévaluer : {len(available)}")
    print(f"k : {k}")
    print("\nChargement du modèle E5...")

    model = SentenceTransformer(
        config["model"],
        revision=str(config["revision"]),
    )

    summary = []

    for exp, collection, original_report in available:

        print(f"\n========== {exp['id']} ==========")

        report = reevaluate_experiment(
            exp,
            collection,
            original_report,
            model,
            questions,
            k,
            config,
        )

        output_path = save_new_report(exp, report)

        metrics = report["metrics"]
        chunk_count = report["experiment"]["chunk_count"]

        print(f"Chunks : {chunk_count}")
        print(f"Precision@{k} : {metrics['precision_at_k']:.4f}")
        print(f"Recall@{k}    : {metrics['recall_at_k']:.4f}")
        print(f"MRR           : {metrics['mrr']:.4f}")
        print(f"Rapport       : {output_path}")

        summary.append({
            "id": exp["id"],
            "chunks": chunk_count,
            **metrics,
        })

    print("\n========== COMPARAISON ==========")

    for result in summary:
        print(
            f"{result['id']} | "
            f"chunks={result['chunks']} | "
            f"P@{k}={result['precision_at_k']:.4f} | "
            f"R@{k}={result['recall_at_k']:.4f} | "
            f"MRR={result['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
