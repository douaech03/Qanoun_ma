
"""Recherche multi-query pour MarocLex AI."""

import argparse

from src.retriever import retrieve


# 1. Décomposer certaines questions comparatives

def generate_subquestions(question):
    """
    Décomposer une comparaison entre la loi 19-12
    et le Code du travail concernant l'âge minimum.

    Pour les autres questions, conserver la question originale.
    """

    text = question.lower()

    if (
        "âge minimum" in text
        and "domestique" in text
        and "code du travail" in text
    ):
        return [
            (
                "Quel est l'âge minimum d'emploi "
                "des travailleurs domestiques selon la loi 19-12 ?"
            ),
            (
                "Quel est l'âge minimum d'admission "
                "des mineurs au travail selon le Code du travail marocain ?"
            ),
        ]

    return [question]


# 2. Répartir les résultats entre les sous-questions

def merge_results(results_by_query, k):
    """Fusionner les résultats en alternant les sous-questions."""

    merged = []
    seen_ids = set()

    max_length = max(
        (len(results) for results in results_by_query),
        default=0,
    )

    for rank in range(max_length):
        for results in results_by_query:

            if rank >= len(results):
                continue

            passage = results[rank]
            passage_id = passage["id"]

            if passage_id in seen_ids:
                continue

            seen_ids.add(passage_id)
            merged.append(passage)

            if len(merged) == k:
                return merged

    return merged


# 3. Retriever multi-query

def multi_query_retrieve(question, k=4):
    """Rechercher les passages pour une ou plusieurs sous-questions."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question ne doit pas être vide.")

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    subquestions = generate_subquestions(question)

    # Une seule question : comportement du retriever habituel
    if len(subquestions) == 1:
        return retrieve(subquestions[0], k=k)

    # Plusieurs questions : recherches indépendantes
    results_by_query = []

    for subquestion in subquestions:
        results = retrieve(subquestion, k=k)
        results_by_query.append(results)

    return merge_results(results_by_query, k)


# 4. Exécution depuis le terminal

def main():

    parser = argparse.ArgumentParser(
        description="Retriever multi-query MarocLex AI"
    )

    parser.add_argument("question", type=str)
    parser.add_argument("--k", type=int, default=4)

    args = parser.parse_args()

    subquestions = generate_subquestions(args.question)

    print("\nSous-questions :")

    for index, question in enumerate(subquestions, start=1):
        print(f"{index}. {question}")

    results = multi_query_retrieve(args.question, k=args.k)

    print(f"\nPassages trouvés : {len(results)}")

    for rank, passage in enumerate(results, start=1):
        print(f"\n--- Passage {rank} ---")
        print(f"Document : {passage['document_id']}")
        print(f"Page     : {passage['page']}")
        print(f"Texte    : {passage['text'][:250]}...")


if __name__ == "__main__":
    main()
