
"""Tests du pipeline d'ingestion MarocLex AI."""

import csv
import hashlib

import pytest

from src import ingest


# --------------------------------------------------
# 1. Test du chargement de la configuration
# --------------------------------------------------

def test_load_config():
    """Vérifier la lecture des paramètres du projet."""

    config = ingest.load_config()

    assert config["embedding"] == "e5"
    assert config["chunk_tokens"] > 0
    assert 0 <= config["overlap"] < config["chunk_tokens"]
    assert config["collection"]
    assert config["model"]
    assert config["revision"]


# --------------------------------------------------
# 2. Test du nettoyage du texte
# --------------------------------------------------

def test_clean_text():
    """Vérifier la suppression des espaces inutiles."""

    texte = "Bonjour     Maroc\n\n\nCode    du travail"

    resultat = ingest.clean_text(texte)

    assert resultat == "Bonjour Maroc\nCode du travail"


# --------------------------------------------------
# 3. Test d'un manifeste valide
# --------------------------------------------------

def test_load_manifest(tmp_path, monkeypatch):
    """Vérifier la lecture des documents déclarés."""

    # Construire un mini-projet temporaire
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    # Créer deux PDF fictifs pour tester les chemins
    (raw / "document1.pdf").write_bytes(b"%PDF-1.4")
    (raw / "document2.pdf").write_bytes(b"%PDF-1.4")

    manifest_path = tmp_path / "manifest.csv"

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "document_id", "path", "source",
            "access", "index_rag"
        ])

        writer.writerow([
            "doc1", "data/raw/document1.pdf",
            "Document public", "public", "true"
        ])

        writer.writerow([
            "doc2", "data/raw/document2.pdf",
            "Document reserve", "restricted", "false"
        ])

    # Rediriger temporairement la racine du projet
    monkeypatch.setattr(ingest, "ROOT", tmp_path)

    documents = ingest.load_manifest(manifest_path)

    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc1"
    assert documents[0]["access"] == "public"


# --------------------------------------------------
# 4. Test d'un manifeste vide
# --------------------------------------------------

def test_empty_manifest(tmp_path):
    """Vérifier le rejet d'un manifeste vide."""

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="vide"):
        ingest.load_manifest(manifest_path)


# --------------------------------------------------
# 5. Test d'un niveau d'accès invalide
# --------------------------------------------------

def test_invalid_access(tmp_path, monkeypatch):
    """Refuser un niveau d'accès inconnu."""

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    (raw / "document.pdf").write_bytes(b"%PDF-1.4")

    manifest_path = tmp_path / "manifest.csv"

    manifest_path.write_text(
        "document_id,path,source,access,index_rag\n"
        "doc1,data/raw/document.pdf,Test,unknown,true\n",
        encoding="utf-8"
    )

    monkeypatch.setattr(ingest, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="Accès invalide"):
        ingest.load_manifest(manifest_path)


# --------------------------------------------------
# 6. Test du découpage en chunks
# --------------------------------------------------

class FakeTokenizer:
    """Tokenizer simplifié utilisé uniquement pour les tests."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)


def test_split_text():
    """Vérifier la taille et le chevauchement des chunks."""

    tokenizer = FakeTokenizer()

    texte = "ABCDEFGHIJ"

    chunks = ingest.split_text(
        texte,
        tokenizer,
        chunk_tokens=6,
        overlap=2
    )

    assert chunks == [
        "ABCDEF",
        "EFGHIJ"
    ]


# --------------------------------------------------
# 7. Test des paramètres de découpage invalides
# --------------------------------------------------

def test_invalid_chunk_parameters():
    """Refuser un overlap supérieur ou égal au chunk."""

    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError):
        ingest.split_text(
            "Bonjour",
            tokenizer,
            chunk_tokens=5,
            overlap=5
        )


# --------------------------------------------------
# 8. Test des métadonnées et des identifiants
# --------------------------------------------------

def test_records_reproducibility(tmp_path, monkeypatch):
    """Vérifier la stabilité des IDs et la traçabilité."""

    monkeypatch.setattr(ingest, "ROOT", tmp_path)

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    pdf_path = raw / "test.pdf"
    pdf_content = b"Document de test"
    pdf_path.write_bytes(pdf_content)

    # Simuler l'extraction sans lire un vrai PDF
    def fake_extract_pages(path):
        return [
            {
                "page": 1,
                "text": "Contrat de travail"
            }
        ]

    monkeypatch.setattr(
        ingest,
        "extract_pages",
        fake_extract_pages
    )

    class FakeModel:
        tokenizer = FakeTokenizer()

    documents = [
        {
            "document_id": "doc_test",
            "source": "Document juridique test",
            "access": "public",
            "pdf_path": pdf_path,
        }
    ]

    config = {
        "corpus_version": "test-v1",
        "chunk_tokens": 100,
        "overlap": 10,
    }

    records1 = ingest.build_records(
        documents, FakeModel(), config
    )

    records2 = ingest.build_records(
        documents, FakeModel(), config
    )

    # Deux exécutions identiques donnent les mêmes IDs
    assert [r["id"] for r in records1] == [
        r["id"] for r in records2
    ]

    assert len(records1) == 1

    metadata = records1[0]["metadata"]

    assert metadata["document_id"] == "doc_test"
    assert metadata["source"] == "Document juridique test"
    assert metadata["page"] == 1
    assert metadata["access"] == "public"
    assert metadata["corpus_version"] == "test-v1"

    # Vérifier l'empreinte du document d'origine
    expected_hash = hashlib.sha256(pdf_content).hexdigest()

    assert metadata["pdf_sha256"] == expected_hash
