
"""Evaluation de la recherche documentaire MarocLex AI."""

import argparse
import json
from pathlib import Path

import yaml

from src.retriever import retrieve


# Racine du projet
ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 1. Chargement des questions d'évaluation
# ==================================================

def load_questions(path=None):
    """
    Charger les questions de référence depuis questions.yaml.

    Les questions en attente de validation sont ignorées.
    """

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

        question_id = item.get("id")
        question = item.get("question")
        relevant = item.get("relevant_passages")

        # Ignorer les questions en attente
        if item.get("status") == "pending":
            continue

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
                f"Passages de référence manquants : {question_id}"
            )

        for passage in relevant:

            if not isinstance(passage, dict):
                raise ValueError(
                    f"Référence invalide : {question_id}"
                )

            if not isinstance(passage.get("document_id"), str):
                raise ValueError(
                    f"document_id invalide : {question_id}"
                )

            page = passage.get("page")

            if (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page <= 0
            ):
                raise ValueError(
                    f"Numéro de page invalide : {question_id}"
                )

        validated_questions.append(item)

    if not validated_questions:
        raise ValueError(
            "Aucune question validée à évaluer."
        )

    return validated_questions


# ==================================================
# 2. Vérification de la pertinence
# ==================================================

def is_relevant(result, expected_passages):
    """
    Vérifier si un résultat correspond à une référence attendue.

    Une correspondance nécessite le même document et la même page.
    """

    return any(
        result.get("document_id") == expected["document_id"]
        and result.get("page") == expected["page"]
        for expected in expected_passages
    )


# ==================================================
# 3. Precision@k
# ==================================================

def precision_at_k(results, expected_passages, k):
    """
    Calculer la précision en comptant les pages uniques.

    Precision@k = références pertinentes uniques retrouvées / k
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    top_results = results[:k]

    if not top_results:
        return 0.0

    retrieved_relevant = {
        (result.get("document_id"), result.get("page"))
        for result in top_results
        if is_relevant(result, expected_passages)
    }

    return len(retrieved_relevant) / k


# ==================================================
# 4. Recall@k
# ==================================================

def recall_at_k(results, expected_passages, k):
    """
    Calculer le rappel sans compter les doublons.

    Recall@k = références pertinentes retrouvées / références attendues
    """

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
        if is_relevant(result, expected_passages)
    }

    found = expected.intersection(retrieved)

    return len(found) / len(expected)


# ==================================================
# 5. Reciprocal Rank
# ==================================================

def reciprocal_rank(results, expected_passages):
    """
    Calculer le Reciprocal Rank d'une question.

    Premier passage pertinent au rang 2 : RR = 1/2.
    Aucun passage pertinent : RR = 0.
    """

    for rank, result in enumerate(results, start=1):

        if is_relevant(result, expected_passages):
            return 1.0 / rank

    return 0.0


# ==================================================
# 6. Evaluation d'une question
# ==================================================

def evaluate_question(item, k):
    """Évaluer une question avec le moteur de recherche."""

    question = item["question"]
    expected = item["relevant_passages"]

    results = retrieve(question, k=k)

    precision = precision_at_k(results, expected, k)
    recall = recall_at_k(results, expected, k)
    rr = reciprocal_rank(results, expected)

    return {
        "id": item["id"],
        "question": question,
        "retrieved_count": len(results),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": rr,
        "results": results,
    }


# ==================================================
# 7. Evaluation de tout le jeu de données
# ==================================================

def evaluate_dataset(questions, k=4):
    """Évaluer toutes les questions et calculer les moyennes."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    if not questions:
        raise ValueError("Aucune question à évaluer.")

    evaluations = []

    for item in questions:

        print(f"Évaluation de {item['id']}...")

        result = evaluate_question(item, k)

        evaluations.append(result)

    count = len(evaluations)

    metrics = {
        "questions_evaluees": count,
        "k": k,

        "precision_at_k": sum(
            item["precision_at_k"]
            for item in evaluations
        ) / count,

        "recall_at_k": sum(
            item["recall_at_k"]
            for item in evaluations
        ) / count,

        "mrr": sum(
            item["reciprocal_rank"]
            for item in evaluations
        ) / count,
    }

    return {
        "metrics": metrics,
        "details": evaluations,
    }


# ==================================================
# 8. Enregistrement du rapport
# ==================================================

def save_report(report, output_path=None):
    """Enregistrer les résultats dans un fichier JSON."""

    if output_path is None:
        output_path = (
            ROOT / "artifacts" / "evaluation" / "report.json"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(f"\nRapport enregistré : {output_path}")


# ==================================================
# 9. Programme principal
# ==================================================

def main():
    """Lancer l'évaluation depuis le terminal."""

    parser = argparse.ArgumentParser(
        description="Évaluation RAG MarocLex AI"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nombre de passages à évaluer.",
    )

    args = parser.parse_args()

    # Charger la configuration
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    k = config["k"] if args.k is None else args.k

    # Charger les questions validées
    questions = load_questions()

    print(
        f"\nNombre de questions évaluées : {len(questions)}"
    )

    print(f"Nombre de passages demandés : {k}\n")

    # Lancer l'évaluation
    report = evaluate_dataset(questions, k)

    # Afficher les métriques
    print("\n========== RÉSULTATS ==========")

    for metric, value in report["metrics"].items():

        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")

        else:
            print(f"{metric}: {value}")

    # Enregistrer le rapport
    save_report(report)


if __name__ == "__main__":
    main()
