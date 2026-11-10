
"""Évaluation de la recherche documentaire MarocLex AI."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.retriever import retrieve


# Racine du projet
ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 1. Charger la configuration
# ==================================================

def load_config():
    """Lire les paramètres de params.yaml."""

    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==================================================
# 2. Charger les questions
# ==================================================

def load_questions(path=None):
    """Charger les questions validées depuis questions.yaml."""

    if path is None:
        path = ROOT / "data" / "evaluation" / "questions.yaml"

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier d'évaluation introuvable : {path}"
        )

    with open(path, encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Structure YAML invalide.")

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError("La liste questions est absente.")

    validated_questions = []
    used_ids = set()

    for item in questions:

        if not isinstance(item, dict):
            raise ValueError("Une question est invalide.")

        # Ignorer les questions non validées
        if item.get("status") == "pending":
            continue

        question_id = item.get("id")
        question = item.get("question")
        relevant = item.get("relevant_passages")

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Identifiant de question invalide.")

        if question_id in used_ids:
            raise ValueError(
                f"Identifiant dupliqué : {question_id}"
            )

        used_ids.add(question_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question vide ou invalide : {question_id}"
            )

        if not isinstance(relevant, list) or not relevant:
            raise ValueError(
                f"Références manquantes : {question_id}"
            )

        for passage in relevant:

            if not isinstance(passage, dict):
                raise ValueError(
                    f"Référence invalide : {question_id}"
                )

            document_id = passage.get("document_id")
            page = passage.get("page")

            if (
                not isinstance(document_id, str)
                or not document_id.strip()
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page <= 0
            ):
                raise ValueError(
                    f"Référence invalide : {question_id}"
                )

        validated_questions.append(item)

    if not validated_questions:
        raise ValueError("Aucune question validée à évaluer.")

    return validated_questions


# ==================================================
# 3. Vérifier la pertinence
# ==================================================

def is_relevant(result, expected_passages):
    """Comparer le document et la page du résultat."""

    return any(
        result.get("document_id") == expected["document_id"]
        and result.get("page") == expected["page"]
        for expected in expected_passages
    )


# ==================================================
# 4. Calculer Precision@k
# ==================================================

def precision_at_k(results, expected_passages, k):
    """Précision sur les références document/page uniques."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    relevant = {
        (result.get("document_id"), result.get("page"))
        for result in results[:k]
        if is_relevant(result, expected_passages)
    }

    return len(relevant) / k


# ==================================================
# 5. Calculer Recall@k
# ==================================================

def recall_at_k(results, expected_passages, k):
    """Rappel sans compter plusieurs fois la même page."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    if not expected_passages:
        return 0.0

    expected = {
        (item["document_id"], item["page"])
        for item in expected_passages
    }

    retrieved = {
        (result.get("document_id"), result.get("page"))
        for result in results[:k]
    }

    return len(expected.intersection(retrieved)) / len(expected)


# ==================================================
# 6. Calculer Reciprocal Rank
# ==================================================

def reciprocal_rank(results, expected_passages):
    """Calculer l'inverse du rang du premier résultat pertinent."""

    for rank, result in enumerate(results, start=1):

        if is_relevant(result, expected_passages):
            return 1.0 / rank

    return 0.0


# ==================================================
# 7. Évaluer une question
# ==================================================

def evaluate_question(item, k):
    """Évaluer les résultats du retriever pour une question."""

    results = retrieve(item["question"], k=k)
    expected = item["relevant_passages"]

    return {
        "id": item["id"],
        "question": item["question"],
        "retrieved_count": len(results),
        "precision_at_k": precision_at_k(
            results, expected, k
        ),
        "recall_at_k": recall_at_k(
            results, expected, k
        ),
        "reciprocal_rank": reciprocal_rank(
            results, expected
        ),
        "results": results,
    }


# ==================================================
# 8. Évaluer toutes les questions
# ==================================================

def evaluate_dataset(questions, k=4):
    """Calculer les métriques moyennes sur les questions."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    if not questions:
        raise ValueError("Aucune question à évaluer.")

    evaluations = []

    for item in questions:

        print(f"Évaluation de {item['id']}...")

        evaluations.append(
            evaluate_question(item, k)
        )

    count = len(evaluations)

    metrics = {
        "questions_evaluees": count,
        "k": k,
        "precision_at_k": sum(
            item["precision_at_k"] for item in evaluations
        ) / count,
        "recall_at_k": sum(
            item["recall_at_k"] for item in evaluations
        ) / count,
        "mrr": sum(
            item["reciprocal_rank"] for item in evaluations
        ) / count,
    }

    return {
        "metrics": metrics,
        "details": evaluations,
    }


# ==================================================
# 9. Ajouter les paramètres de l'expérience
# ==================================================

def add_experiment_info(report, config, k):
    """Enregistrer la configuration utilisée pour l'évaluation."""

    report["experiment"] = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_version": str(config["corpus_version"]),
        "model": config["model"],
        "model_revision": str(config["revision"]),
        "chunk_tokens": int(config["chunk_tokens"]),
        "overlap": int(config["overlap"]),
        "k": k,
        "collection": config["collection"],
        "index_dir": config["index_dir"],
        "max_distance": config.get("max_distance"),
        "evaluation_level": "document_page",
    }

    return report


# ==================================================
# 10. Sauvegarder le rapport JSON
# ==================================================

def save_report(report, output_path=None):
    """Sauvegarder un rapport sans écraser une expérience existante."""

    if output_path is None:

        k = report["metrics"]["k"]

        output_path = (
            ROOT / "artifacts" / "evaluation"
            / f"report_k{k}.json"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ne pas écraser un rapport existant.
    # Créer un nom numéroté si nécessaire.
    if output_path.exists():

        base_path = output_path
        number = 2

        while output_path.exists():
            output_path = base_path.with_name(
                f"{base_path.stem}_{number}{base_path.suffix}"
            )
            number += 1

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(f"\nRapport enregistré : {output_path}")

    return output_path


# ==================================================
# 11. Programme principal
# ==================================================

def main():
    """Lancer l'évaluation depuis PowerShell."""

    parser = argparse.ArgumentParser(
        description="Évaluation documentaire MarocLex AI"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nombre de chunks recherchés",
    )

    args = parser.parse_args()

    config = load_config()

    k = int(config["k"]) if args.k is None else args.k

    questions = load_questions()

    print(f"\nQuestions évaluées : {len(questions)}")
    print(f"Configuration : k={k}")
    print(
        f"Chunk size : {config['chunk_tokens']} | "
        f"Overlap : {config['overlap']}\n"
    )

    report = evaluate_dataset(questions, k=k)

    report = add_experiment_info(
        report,
        config,
        k,
    )

    print("\n========== RÉSULTATS ==========")

    for metric, value in report["metrics"].items():

        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

        else:
            print(f"{metric}: {value}")

    save_report(report)


if __name__ == "__main__":
    main()
