
"""Comparaison des configurations de chunking de MarocLex AI."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from src import ingest, evaluate


ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 1. Charger les configurations expérimentales
# ==================================================

def load_experiments():
    """Lire et vérifier les expériences E1 à E5."""

    path = ROOT / "configs" / "chunking_experiments.yaml"

    with open(path, encoding="utf-8-sig") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Configuration expérimentale invalide.")

    k = data.get("k")
    experiments = data.get("experiments")

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Aucune expérience définie.")

    used_ids = set()

    for exp in experiments:
        if not isinstance(exp, dict):
            raise ValueError("Expérience invalide.")

        exp_id = exp.get("id")
        size = exp.get("chunk_tokens")
        overlap = exp.get("overlap")

        if not isinstance(exp_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", exp_id
        ):
            raise ValueError("Identifiant d'expérience invalide.")

        if exp_id in used_ids:
            raise ValueError(f"ID dupliqué : {exp_id}")

        used_ids.add(exp_id)

        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or not 0 <= overlap < size
        ):
            raise ValueError(
                f"Paramètres de chunking invalides : {exp_id}"
            )

    return k, experiments


# ==================================================
# 2. Calculer une empreinte SHA-256
# ==================================================

def sha256_file(path):
    """Identifier le contenu d'un fichier par son empreinte."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# ==================================================
# 3. Construire l'index d'une expérience
# ==================================================

def create_experiment_index(exp, config, documents, model):
    """Créer un index isolé sans modifier l'index principal."""

    exp_dir = ROOT / "artifacts" / "experiments" / exp["id"]
    index_dir = exp_dir / "index"

    if exp_dir.exists():
        raise FileExistsError(
            f"Expérience déjà présente : {exp_dir}. "
            "Aucune donnée existante ne sera remplacée."
        )

    # Copier la configuration, puis adapter le chunking
    exp_config = config.copy()
    exp_config["chunk_tokens"] = exp["chunk_tokens"]
    exp_config["overlap"] = exp["overlap"]

    # Réutiliser le pipeline d'ingestion existant
    records = ingest.build_records(
        documents,
        model,
        exp_config,
    )

    print(f"Chunks générés : {len(records)}")

    index_dir.mkdir(parents=True, exist_ok=False)

    client = chromadb.PersistentClient(
        path=str(index_dir)
    )

    collection = client.create_collection(
        name=config["collection"],
        metadata={"hnsw:space": "cosine"},
    )

    # Calculer et enregistrer les embeddings par lots
    batch_size = 32

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        texts = [
            "passage: " + item["text"]
            for item in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings.tolist(),
        )

    if collection.count() != len(records):
        raise RuntimeError("Nombre de chunks indexés incorrect.")

    return collection, len(records), exp_dir


# ==================================================
# 4. Rechercher dans l'index expérimental
# ==================================================

def search_experiment(question, collection, model, k, config):
    """Effectuer une recherche publique dans l'index expérimental."""

    query_vector = model.encode(
        ["query: " + question.strip()],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=k,
        where={"access": "public"},
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    max_distance = config.get("max_distance")

    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if max_distance is not None:
            if distance > float(max_distance):
                continue

        passages.append({
            "id": chunk_id,
            "text": text,
            "document_id": metadata["document_id"],
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": float(distance),
            "access": metadata["access"],
        })

    return passages


# ==================================================
# 5. Évaluer une configuration
# ==================================================

def evaluate_experiment(collection, model, questions, k, config):
    """Calculer les métriques sur les questions validées."""

    details = []

    for item in questions:
        results = search_experiment(
            item["question"],
            collection,
            model,
            k,
            config,
        )

        expected = item["relevant_passages"]

        details.append({
            "id": item["id"],
            "question": item["question"],
            "retrieved_count": len(results),
            "precision_at_k": evaluate.precision_at_k(
                results, expected, k
            ),
            "recall_at_k": evaluate.recall_at_k(
                results, expected, k
            ),
            "reciprocal_rank": evaluate.reciprocal_rank(
                results, expected
            ),
            "retrieved_passages": [
                {
                    "document_id": result["document_id"],
                    "page": result["page"],
                    "distance": result["distance"],
                }
                for result in results
            ],
        })

    count = len(details)

    return {
        "questions_evaluees": count,
        "precision_at_k": sum(
            item["precision_at_k"] for item in details
        ) / count,
        "recall_at_k": sum(
            item["recall_at_k"] for item in details
        ) / count,
        "mrr": sum(
            item["reciprocal_rank"] for item in details
        ) / count,
    }, details


# ==================================================
# 6. Exécuter une expérience complète
# ==================================================

def run_experiment(exp, k, config, documents, model, questions):
    """Construire, évaluer et enregistrer une expérience."""

    print(f"\n========== {exp['id']} ==========")
    print(f"Chunk size : {exp['chunk_tokens']}")
    print(f"Overlap : {exp['overlap']}")
    print(f"k : {k}")

    collection, chunk_count, exp_dir = create_experiment_index(
        exp, config, documents, model
    )

    metrics, details = evaluate_experiment(
        collection, model, questions, k, config
    )

    report = {
        "experiment": {
            "id": exp["id"],
            "name": exp.get("name", exp["id"]),
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "chunk_tokens": exp["chunk_tokens"],
            "overlap": exp["overlap"],
            "k": k,
            "model": config["model"],
            "revision": str(config["revision"]),
            "corpus_version": str(config["corpus_version"]),
            "collection": config["collection"],
            "max_distance": config.get("max_distance"),
            "evaluation_level": "document_page",
            "index_path": str(exp_dir / "index"),
            "chunk_count": chunk_count,
            "document_hashes": {
                doc["document_id"]: sha256_file(doc["pdf_path"])
                for doc in documents
            },
            "questions_sha256": sha256_file(
                ROOT / "data" / "evaluation" / "questions.yaml"
            ),
        },
        "metrics": metrics,
        "details": details,
    }

    report_path = exp_dir / "report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Precision@{k} : {metrics['precision_at_k']:.4f}")
    print(f"Recall@{k}    : {metrics['recall_at_k']:.4f}")
    print(f"MRR           : {metrics['mrr']:.4f}")
    print(f"Rapport       : {report_path}")

    return report


# ==================================================
# 7. Programme principal
# ==================================================

def main():
    """Exécuter une ou plusieurs expériences depuis PowerShell."""

    parser = argparse.ArgumentParser(
        description="Comparaison des configurations de chunking"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Identifiant d'une seule expérience, par exemple E1",
    )

    args = parser.parse_args()

    k, experiments = load_experiments()

    if args.experiment is not None:
        experiments = [
            exp for exp in experiments
            if exp["id"] == args.experiment
        ]

        if not experiments:
            parser.error("Identifiant d'expérience inconnu.")

    # Vérification avant toute indexation
    for exp in experiments:
        exp_dir = ROOT / "artifacts" / "experiments" / exp["id"]

        if exp_dir.exists():
            raise FileExistsError(
                f"{exp_dir} existe déjà. "
                "Arrêt pour protéger les expériences existantes."
            )

    config = ingest.load_config()

    documents = ingest.load_manifest(
        ROOT / config["manifest"]
    )

    questions = evaluate.load_questions()

    print(f"Documents : {len(documents)}")
    print(f"Questions : {len(questions)}")
    print(f"Expériences : {len(experiments)}")

    print("\nChargement du modèle E5...")

    model = SentenceTransformer(
        config["model"],
        revision=str(config["revision"]),
    )

    summaries = []

    for exp in experiments:
        report = run_experiment(
            exp, k, config, documents, model, questions
        )

        summaries.append({
            "id": exp["id"],
            "chunk_tokens": exp["chunk_tokens"],
            "overlap": exp["overlap"],
            "chunks": report["experiment"]["chunk_count"],
            **report["metrics"],
        })

    print("\n========== COMPARAISON ==========")

    for summary in summaries:
        print(
            f"{summary['id']} | "
            f"chunks={summary['chunks']} | "
            f"P@{k}={summary['precision_at_k']:.4f} | "
            f"R@{k}={summary['recall_at_k']:.4f} | "
            f"MRR={summary['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
