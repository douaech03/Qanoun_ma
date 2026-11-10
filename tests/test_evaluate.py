
"""Tests du module d'évaluation de MarocLex AI."""

import json

import pytest

from src import evaluate


# ==================================================
# 1. Données fictives réutilisables
# ==================================================

@pytest.fixture
def sample_results():
    """Simuler quatre passages retournés par le retriever."""

    return [
        {
            "id": "chunk1",
            "document_id": "loi_19_12",
            "page": 6,
            "distance": 0.10,
        },
        {
            "id": "chunk2",
            "document_id": "loi_19_12",
            "page": 4,
            "distance": 0.12,
        },
        {
            "id": "chunk3",
            "document_id": "loi_19_12",
            "page": 6,
            "distance": 0.14,
        },
        {
            "id": "chunk4",
            "document_id": "code_travail",
            "page": 10,
            "distance": 0.16,
        },
    ]


@pytest.fixture
def expected_passages():
    """Définir une page pertinente attendue."""

    return [
        {
            "document_id": "loi_19_12",
            "page": 6,
        }
    ]


# ==================================================
# 2. Test de la pertinence
# ==================================================

def test_is_relevant(sample_results, expected_passages):
    """Vérifier la correspondance document + page."""

    assert evaluate.is_relevant(
        sample_results[0], expected_passages
    )

    assert not evaluate.is_relevant(
        sample_results[1], expected_passages
    )


# ==================================================
# 3. Test de Precision@k
# ==================================================

def test_precision_at_k(sample_results, expected_passages):
    """Vérifier la précision sans double comptage."""

    precision = evaluate.precision_at_k(
        sample_results,
        expected_passages,
        k=4,
    )

    # Deux chunks de la page 6, mais une seule
    # référence pertinente unique.
    assert precision == pytest.approx(0.25)


# ==================================================
# 4. Test de Recall@k
# ==================================================

def test_recall_at_k(sample_results, expected_passages):
    """Vérifier que le rappel ne dépasse jamais 1."""

    recall = evaluate.recall_at_k(
        sample_results,
        expected_passages,
        k=4,
    )

    assert recall == pytest.approx(1.0)
    assert 0 <= recall <= 1


# ==================================================
# 5. Test du Reciprocal Rank
# ==================================================

def test_reciprocal_rank(
    sample_results,
    expected_passages,
):
    """Vérifier le rang du premier passage pertinent."""

    rr = evaluate.reciprocal_rank(
        sample_results,
        expected_passages,
    )

    assert rr == pytest.approx(1.0)


def test_reciprocal_rank_second_position():
    """Vérifier le cas où le bon passage est deuxième."""

    results = [
        {"document_id": "doc1", "page": 1},
        {"document_id": "doc2", "page": 2},
        {"document_id": "doc3", "page": 3},
    ]

    expected = [
        {"document_id": "doc2", "page": 2}
    ]

    rr = evaluate.reciprocal_rank(results, expected)

    assert rr == pytest.approx(0.5)


# ==================================================
# 6. Test sans passage pertinent
# ==================================================

def test_no_relevant_result(sample_results):
    """Vérifier les métriques lorsqu'aucun résultat n'est pertinent."""

    expected = [
        {
            "document_id": "inconnu",
            "page": 99,
        }
    ]

    assert evaluate.precision_at_k(
        sample_results, expected, 4
    ) == 0.0

    assert evaluate.recall_at_k(
        sample_results, expected, 4
    ) == 0.0

    assert evaluate.reciprocal_rank(
        sample_results, expected
    ) == 0.0


# ==================================================
# 7. Test d'un k invalide
# ==================================================

@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(sample_results, expected_passages, k):
    """Refuser les valeurs invalides de k."""

    with pytest.raises(ValueError):
        evaluate.precision_at_k(
            sample_results, expected_passages, k
        )

    with pytest.raises(ValueError):
        evaluate.recall_at_k(
            sample_results, expected_passages, k
        )


# ==================================================
# 8. Test du chargement des questions YAML
# ==================================================

def test_load_questions(tmp_path):
    """Vérifier le chargement et l'exclusion des questions pending."""

    yaml_file = tmp_path / "questions.yaml"

    yaml_file.write_text(
        "questions:\n"
        "  - id: Q001\n"
        "    question: Question valide ?\n"
        "    relevant_passages:\n"
        "      - document_id: doc1\n"
        "        page: 1\n"
        "  - id: Q002\n"
        "    question: Question en attente ?\n"
        "    relevant_passages: []\n"
        "    status: pending\n",
        encoding="utf-8",
    )

    questions = evaluate.load_questions(yaml_file)

    assert len(questions) == 1
    assert questions[0]["id"] == "Q001"


# ==================================================
# 9. Test d'un YAML invalide
# ==================================================

def test_invalid_yaml(tmp_path):
    """Refuser un fichier YAML sans questions."""

    yaml_file = tmp_path / "questions.yaml"

    yaml_file.write_text(
        "version: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        evaluate.load_questions(yaml_file)


# ==================================================
# 10. Test d'évaluation d'une question
# ==================================================

def test_evaluate_question(
    monkeypatch,
    sample_results,
):
    """Évaluer une question sans utiliser le vrai retriever."""

    # Remplacer temporairement la fonction retrieve()
    monkeypatch.setattr(
        evaluate,
        "retrieve",
        lambda question, k: sample_results[:k],
    )

    item = {
        "id": "Q_TEST",
        "question": "Quel est l'âge minimum ?",
        "relevant_passages": [
            {
                "document_id": "loi_19_12",
                "page": 6,
            }
        ],
    }

    result = evaluate.evaluate_question(item, k=4)

    assert result["id"] == "Q_TEST"
    assert result["retrieved_count"] == 4
    assert result["precision_at_k"] == pytest.approx(0.25)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["reciprocal_rank"] == pytest.approx(1.0)


# ==================================================
# 11. Test de l'évaluation globale
# ==================================================

def test_evaluate_dataset(monkeypatch):
    """Vérifier le calcul des moyennes."""

    def fake_evaluate_question(item, k):
        return {
            "id": item["id"],
            "question": item["question"],
            "retrieved_count": k,
            "precision_at_k": item["precision"],
            "recall_at_k": item["recall"],
            "reciprocal_rank": item["rr"],
            "results": [],
        }

    monkeypatch.setattr(
        evaluate,
        "evaluate_question",
        fake_evaluate_question,
    )

    questions = [
        {
            "id": "Q001",
            "question": "Question 1",
            "precision": 0.25,
            "recall": 1.0,
            "rr": 1.0,
        },
        {
            "id": "Q002",
            "question": "Question 2",
            "precision": 0.0,
            "recall": 0.0,
            "rr": 0.0,
        },
    ]

    report = evaluate.evaluate_dataset(questions, k=4)

    metrics = report["metrics"]

    assert metrics["questions_evaluees"] == 2
    assert metrics["precision_at_k"] == pytest.approx(0.125)
    assert metrics["recall_at_k"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)


# ==================================================
# 12. Test de sauvegarde du rapport JSON
# ==================================================

def test_save_report(tmp_path):
    """Vérifier la création du rapport d'évaluation."""

    output_path = tmp_path / "evaluation" / "report.json"

    report = {
        "metrics": {
            "questions_evaluees": 2,
            "k": 4,
            "precision_at_k": 0.25,
            "recall_at_k": 1.0,
            "mrr": 1.0,
        },
        "details": [],
    }

    evaluate.save_report(report, output_path)

    assert output_path.exists()

    with open(output_path, encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report == report
