
"""Tests de la recherche documentaire MarocLex AI."""

import pytest

from src import retriever


# ==================================================
# 1. Faux modèle E5
# ==================================================

class FakeVector(list):
    """Vecteur fictif compatible avec .tolist()."""

    def tolist(self):
        return list(self)


class FakeModel:
    """Simuler E5 sans charger le vrai modèle."""

    def __init__(self):
        self.last_text = None

    def encode(self, texts, normalize_embeddings=False):
        self.last_text = texts[0]

        return [FakeVector([0.1, 0.2, 0.3])]


# ==================================================
# 2. Fausse collection ChromaDB
# ==================================================

class FakeCollection:
    """Simuler une collection contenant deux documents."""

    def __init__(self):
        self.last_query = None

        self.records = [
            {
                "id": "chunk_public",
                "text": "Le salarié bénéficie de droits.",
                "distance": 0.12,
                "metadata": {
                    "document_id": "code_travail",
                    "source": "Code du travail marocain",
                    "page": 10,
                    "access": "public",
                    "file": "data/raw/code_travail.pdf",
                    "corpus_version": "test-v1",
                },
            },
            {
                "id": "chunk_restricted",
                "text": "Note interne fictive réservée.",
                "distance": 0.08,
                "metadata": {
                    "document_id": "note_interne",
                    "source": "Document interne fictif",
                    "page": 1,
                    "access": "restricted",
                    "file": "data/synthetic/note.pdf",
                    "corpus_version": "test-v1",
                },
            },
        ]

    def query(
        self,
        query_embeddings,
        n_results,
        where,
        include,
    ):
        """Simuler le filtrage des métadonnées ChromaDB."""

        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }

        access_condition = where["access"]

        if isinstance(access_condition, str):
            allowed = {access_condition}
        else:
            allowed = set(access_condition["$in"])

        filtered = [
            record for record in self.records
            if record["metadata"]["access"] in allowed
        ]

        # Trier par distance croissante
        filtered.sort(key=lambda item: item["distance"])
        filtered = filtered[:n_results]

        return {
            "ids": [[item["id"] for item in filtered]],
            "documents": [[item["text"] for item in filtered]],
            "metadatas": [[item["metadata"] for item in filtered]],
            "distances": [[item["distance"] for item in filtered]],
        }


# ==================================================
# 3. Préparer un environnement de test
# ==================================================

@pytest.fixture
def fake_environment(monkeypatch):
    """Remplacer temporairement E5 et ChromaDB."""

    config = {
        "model": "intfloat/multilingual-e5-small",
        "revision": "test-revision",
        "collection": "maroclex",
        "index_dir": "artifacts/index",
        "k": 4,
        "max_distance": None,
    }

    model = FakeModel()
    collection = FakeCollection()

    monkeypatch.setattr(
        retriever,
        "load_config",
        lambda: config,
    )

    monkeypatch.setattr(
        retriever,
        "load_model",
        lambda model_name, revision: model,
    )

    monkeypatch.setattr(
        retriever,
        "get_collection",
        lambda settings: collection,
    )

    return config, model, collection


# ==================================================
# 4. Test : recherche publique
# ==================================================

def test_public_access(fake_environment):
    """Un utilisateur public ne voit pas les documents réservés."""

    config, model, collection = fake_environment

    passages = retriever.retrieve(
        "Quels sont les droits du salarié ?"
    )

    assert len(passages) == 1
    assert passages[0]["access"] == "public"

    assert collection.last_query["where"] == {
        "access": "public"
    }

    assert all(
        passage["access"] != "restricted"
        for passage in passages
    )


# ==================================================
# 5. Test : recherche avec autorisation
# ==================================================

def test_authorized_access(fake_environment):
    """Vérifier le filtre multi-niveaux d'accès."""

    passages = retriever.retrieve(
        "Quels sont les droits du salarié ?",
        allowed_access={"public", "restricted"},
    )

    assert len(passages) == 2

    access_types = {
        passage["access"] for passage in passages
    }

    assert access_types == {"public", "restricted"}


# ==================================================
# 6. Test : nombre de résultats k
# ==================================================

def test_top_k(fake_environment):
    """Vérifier que k est transmis à ChromaDB."""

    config, model, collection = fake_environment

    passages = retriever.retrieve(
        "Contrat de travail",
        k=1,
        allowed_access={"public", "restricted"},
    )

    assert collection.last_query["n_results"] == 1
    assert len(passages) == 1


# ==================================================
# 7. Test : question vide
# ==================================================

@pytest.mark.parametrize(
    "question",
    ["", "   ", None],
)
def test_empty_question(fake_environment, question):
    """Refuser une question vide ou invalide."""

    with pytest.raises(ValueError, match="vide"):
        retriever.retrieve(question)


# ==================================================
# 8. Test : valeur k invalide
# ==================================================

@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(fake_environment, k):
    """Refuser un nombre de résultats invalide."""

    with pytest.raises(ValueError):
        retriever.retrieve(
            "Contrat de travail",
            k=k,
        )


# ==================================================
# 9. Test : métadonnées et traçabilité
# ==================================================

def test_metadata(fake_environment):
    """Vérifier les informations associées aux passages."""

    passages = retriever.retrieve(
        "Droits du salarié"
    )

    assert passages

    passage = passages[0]

    required_fields = {
        "id",
        "text",
        "source",
        "document_id",
        "page",
        "distance",
        "access",
        "file",
        "corpus_version",
    }

    assert required_fields.issubset(passage.keys())

    assert passage["source"] == "Code du travail marocain"
    assert passage["page"] == 10
    assert isinstance(passage["distance"], float)


# ==================================================
# 10. Test : distance maximale
# ==================================================

def test_max_distance(fake_environment):
    """Exclure les passages au-dessus du seuil."""

    config, model, collection = fake_environment

    config["max_distance"] = 0.10

    passages = retriever.retrieve(
        "Contrat de travail",
        allowed_access={"public", "restricted"},
    )

    assert len(passages) == 1
    assert passages[0]["distance"] == pytest.approx(0.08)


# ==================================================
# 11. Test : utilisation du préfixe E5
# ==================================================

def test_e5_query_prefix(fake_environment):
    """Vérifier le format des questions pour E5."""

    config, model, collection = fake_environment

    retriever.retrieve(
        "Quels sont les droits du salarié ?"
    )

    assert model.last_text == (
        "query: Quels sont les droits du salarié ?"
    )

    assert collection.last_query["query_embeddings"] == [
        [0.1, 0.2, 0.3]
    ]


# ==================================================
# 12. Test : accès invalide
# ==================================================

def test_invalid_access(fake_environment):
    """Refuser les niveaux d'accès inconnus."""

    with pytest.raises(ValueError, match="accès invalides"):
        retriever.retrieve(
            "Contrat de travail",
            allowed_access={"admin"},
        )
