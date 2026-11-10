
"""Comparaison des retrievers de MarocLex AI."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluate import (
    load_questions,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retriever import retrieve
from src.hybrid_retriever import hybrid_retrieve
from src.multi_query_retriever import multi_query_retrieve


ROOT = Path(__file__).resolve().parent.parent

METHODS = {
    "e5": retrieve,
    "hybrid": hybrid_retrieve,
    "multi_query": multi_query_retrieve,
}


# 1. Évaluer une méthode

def evaluate_method(name, retriever, questions, k):
    """Calculer les métriques pour un retriever."""

    details = []

    for item in questions:
        results = retriever(item["question"], k=k)
        expected = item["relevant_passages"]

        details.append({
            "id": item["id"],
            "precision_at_k": precision_at_k(
                results, expected, k
            ),
            "recall_at_k": recall_at_k(
                results, expected, k
            ),
            "reciprocal_rank": reciprocal_rank(
                results, expected
            ),
            "retrieved_passages": [
                {
                    "document_id": result["document_id"],
                    "page": result["page"],
                }
                for result in results
            ],
        })

    n = len(details)

    return {
        "method": name,
        "questions_evaluees": n,
        "k": k,
        "precision_at_k": sum(
            d["precision_at_k"] for d in details
        ) / n,
        "recall_at_k": sum(
            d["recall_at_k"] for d in details
        ) / n,
        "mrr": sum(
            d["reciprocal_rank"] for d in details
        ) / n,
        "details": details,
    }


# 2. Sauvegarder la comparaison

def save_comparison(reports, k):
    """Enregistrer les résultats sans écraser un ancien rapport."""

    output_dir = ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = (
        output_dir / f"retrievers_k{k}_{timestamp}.json"
    )

    with open(output_path, "x", encoding="utf-8") as file:
        json.dump(
            {
                "date_utc": datetime.now(timezone.utc).isoformat(),
                "k": k,
                "reports": reports,
            },
            file,
            ensure_ascii=False,
            indent=4,
        )

    return output_path


# 3. Programme principal

def main():

    parser = argparse.ArgumentParser(
        description="Comparer les retrievers MarocLex AI"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("k doit être positif.")

    questions = load_questions()

    reports = []

    print(f"\nQuestions : {len(questions)}")
    print(f"k : {args.k}")

    for name, retriever in METHODS.items():

        print(f"\nÉvaluation : {name}")

        report = evaluate_method(
            name,
            retriever,
            questions,
            args.k,
        )

        reports.append(report)

        print(
            f"P@{args.k} : {report['precision_at_k']:.4f} | "
            f"R@{args.k} : {report['recall_at_k']:.4f} | "
            f"MRR : {report['mrr']:.4f}"
        )

    print("\n========== COMPARAISON ==========")

    for report in reports:
        print(
            f"{report['method']:12} | "
            f"P@{args.k}={report['precision_at_k']:.4f} | "
            f"R@{args.k}={report['recall_at_k']:.4f} | "
            f"MRR={report['mrr']:.4f}"
        )

    output_path = save_comparison(reports, args.k)

    print(f"\nRapport enregistré : {output_path}")


if __name__ == "__main__":
    main()
