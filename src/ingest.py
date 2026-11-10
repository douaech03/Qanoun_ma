
"""Ingestion des documents juridiques dans ChromaDB."""

import argparse
import csv
import hashlib
import re
from pathlib import Path

import chromadb
import pymupdf as fitz
import yaml
from sentence_transformers import SentenceTransformer


# Racine du projet MarocLex AI
ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Lire les paramètres existants dans params.yaml."""
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_manifest(manifest_path):
    """Lire et vérifier la liste des documents à indexer."""
    required = {"document_id", "path", "source", "access"}

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest_path}")

    with open(manifest_path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("Le manifeste est vide.")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

        documents = list(reader)

    if not documents:
        raise ValueError("Le manifeste ne contient aucun document.")

    ids = set()
    selected = []

    for doc in documents:
        if not all(doc.get(field, "").strip() for field in required):
            raise ValueError("Métadonnées obligatoires manquantes.")

        doc_id = doc["document_id"].strip()

        if doc_id in ids:
            raise ValueError(f"Identifiant dupliqué : {doc_id}")
        ids.add(doc_id)

        access = doc["access"].strip().lower()
        if access not in {"public", "restricted"}:
            raise ValueError(f"Accès invalide pour {doc_id} : {access}")

        index_rag = doc.get("index_rag", "true").strip().lower()
        if index_rag not in {"true", "false"}:
            raise ValueError(f"index_rag invalide pour {doc_id}")

        if index_rag == "false":
            continue

        pdf_path = (ROOT / doc["path"].strip()).resolve()
        raw_dir = (ROOT / "data" / "raw").resolve()

        if not pdf_path.is_relative_to(raw_dir):
            raise ValueError(f"PDF hors de data/raw : {pdf_path}")

        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        doc["document_id"] = doc_id
        doc["access"] = access
        doc["pdf_path"] = pdf_path
        selected.append(doc)

    if not selected:
        raise ValueError("Aucun document sélectionné pour l'indexation.")

    return selected


def clean_text(text):
    """Supprimer les espaces inutiles sans changer les mots."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    return text.strip()


def extract_pages(pdf_path):
    """Extraire le texte et le numéro de chaque page."""
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not pages:
        raise ValueError(
            f"Aucun texte extrait de {pdf_path.name}. "
            "Le PDF est peut-être scanné."
        )

    return pages


def split_text(text, tokenizer, chunk_tokens, overlap):
    """Découper un texte en chunks avec chevauchement."""
    if chunk_tokens <= 0 or not 0 <= overlap < chunk_tokens:
        raise ValueError("Paramètres de découpage invalides.")

    tokens = tokenizer.encode(
    text,
    add_special_tokens=False,
    truncation=False,
    verbose=False
    )
    step = chunk_tokens - overlap
    chunks = []

    for start in range(0, len(tokens), step):
        selected_tokens = tokens[start:start + chunk_tokens]

        if not selected_tokens:
            break

        chunk = tokenizer.decode(
            selected_tokens,
            skip_special_tokens=True
        ).strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_tokens >= len(tokens):
            break

    return chunks


def build_records(documents, model, config):
    """Préparer les passages, identifiants et métadonnées."""
    records = []
    corpus_version = str(config["corpus_version"])

    for doc in documents:
        pdf_path = doc["pdf_path"]
        pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        pages = extract_pages(pdf_path)

        for page in pages:
            chunks = split_text(
                page["text"],
                model.tokenizer,
                int(config["chunk_tokens"]),
                int(config["overlap"]),
            )

            for chunk_number, chunk in enumerate(chunks, start=1):
                content = (
                    f"{corpus_version}|{doc['document_id']}|"
                    f"{page['page']}|{chunk_number}|{chunk}"
                )
                chunk_id = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                records.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "document_id": doc["document_id"],
                        "source": doc["source"].strip(),
                        "file": str(pdf_path.relative_to(ROOT)),
                        "page": page["page"],
                        "chunk_number": chunk_number,
                        "access": doc["access"],
                        "corpus_version": corpus_version,
                        "pdf_sha256": pdf_hash,
                    },
                })

    if not records:
        raise ValueError("Aucun passage généré.")

    return records


def build_index(rebuild=False):
    """Créer l'index vectoriel unique de MarocLex AI."""
    config = load_config()

    manifest_path = ROOT / config["manifest"]
    documents = load_manifest(manifest_path)

    print(f"Documents sélectionnés : {len(documents)}")
    print("Chargement du modèle E5...")

    model = SentenceTransformer(
        config["model"],
        revision=str(config["revision"]),
    )

    records = build_records(documents, model, config)
    print(f"Passages générés : {len(records)}")

    print("Calcul des embeddings...")
    texts = [record["text"] for record in records]

    embeddings = model.encode(
        ["passage: " + text for text in texts],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    index_path = ROOT / config["index_dir"]
    index_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_path))
    collection_name = config["collection"]

    existing = [
        col.name for col in client.list_collections()
    ]

    if collection_name in existing:
        old_collection = client.get_collection(collection_name)

        if old_collection.count() > 0 and not rebuild:
            raise ValueError(
                "L'index existe déjà. Utilise --rebuild "
                "pour le reconstruire volontairement."
            )

        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(records), 100):
        batch = records[start:start + 100]
        batch_embeddings = embeddings[start:start + 100]

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
            embeddings=batch_embeddings.tolist(),
        )

    print(f"Index enregistré dans : {index_path}")
    print(f"Collection : {collection.name}")
    print(f"Total de passages : {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remplacer volontairement l'index existant.",
    )
    args = parser.parse_args()

    build_index(rebuild=args.rebuild)
