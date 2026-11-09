
"""Recherche sémantique dans l'index juridique MarocLex AI."""

import argparse
from functools import lru_cache
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


# Racine du projet
ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Charger les paramètres depuis params.yaml."""
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=1)
def load_model(model_name, revision):
    """Charger le modèle E5 une seule fois par configuration."""
    return SentenceTransformer(
        model_name,
        revision=str(revision),
    )


def get_collection(config):
    """Ouvrir la collection ChromaDB déjà construite."""
    index_path = ROOT / config["index_dir"]

    if not index_path.exists():
        raise FileNotFoundError(
            "Index introuvable. Lance d'abord : python -m src.ingest"
        )

    client = chromadb.PersistentClient(path=str(index_path))

    try:
        collection = client.get_collection(
            name=config["collection"]
        )
    except Exception as error:
        raise RuntimeError(
            "Collection ChromaDB introuvable. "
            "Vérifie l'indexation."
        ) from error

    return collection


def retrieve(question, k=None, allowed_access=None):
    """
    Rechercher les passages juridiques pertinents.

    Paramètres
    ----------
    question : str
        Question posée par l'utilisateur.

    k : int ou None
        Nombre maximum de passages à retourner.
        None utilise la valeur de params.yaml.

    allowed_access : collection de str ou None
        Niveaux d'accès autorisés par l'application.
        Par défaut, seuls les documents publics sont accessibles.

    Retour
    ------
    list[dict]
        Passages avec texte, source, page, distance
        et métadonnées.
    """

    config = load_config()

    # 1. Vérifier la question
    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    # 2. Lire le nombre de résultats souhaité
    if k is None:
        k = int(config["k"])

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k doit être un entier positif.")

    # 3. Définir les accès autorisés
    # Par défaut : documents publics uniquement.
    if allowed_access is None:
        allowed_access = {"public"}

    allowed_access = set(allowed_access)

    valid_access = {"public", "restricted"}

    if not allowed_access or not allowed_access.issubset(valid_access):
        raise ValueError("Niveaux d'accès invalides.")

    # 4. Ouvrir l'index existant
    collection = get_collection(config)

    # 5. Charger le même modèle E5 que pour l'ingestion
    model = load_model(
        config["model"],
        str(config["revision"]),
    )

    # 6. Transformer la question en vecteur
    query_embedding = model.encode(
        ["query: " + question.strip()],
        normalize_embeddings=True,
    )[0]

    # 7. Appliquer le filtre d'accès documentaire
    if len(allowed_access) == 1:
        access_filter = {
            "access": next(iter(allowed_access))
        }
    else:
        access_filter = {
            "access": {"$in": sorted(allowed_access)}
        }

    # 8. Recherche vectorielle top-k dans ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        where=access_filter,
        include=["documents", "metadatas", "distances"],
    )

    # 9. Construire les résultats avec leurs sources
    passages = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    identifiers = results["ids"][0]

    max_distance = config.get("max_distance")

    for doc_id, text, metadata, distance in zip(
        identifiers,
        documents,
        metadatas,
        distances,
    ):
        # Filtrage supplémentaire selon la distance
        if max_distance is not None:
            if distance > float(max_distance):
                continue

        passage = {
            "id": doc_id,
            "text": text,
            "source": metadata.get("source"),
            "document_id": metadata.get("document_id"),
            "page": metadata.get("page"),
            "distance": float(distance),
            "access": metadata.get("access"),
            "file": metadata.get("file"),
            "corpus_version": metadata.get("corpus_version"),
        }

        passages.append(passage)

    return passages


def main():
    """Tester la recherche depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Recherche documentaire MarocLex AI"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question juridique à rechercher",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Nombre de passages à retourner",
    )

    args = parser.parse_args()

    # Le mode terminal reste limité aux documents publics.
    passages = retrieve(
        question=args.question,
        k=args.k,
    )

    print(f"\nNombre de passages trouvés : {len(passages)}")

    for index, passage in enumerate(passages, start=1):
        print(f"\n--- Passage {index} ---")
        print(f"Source   : {passage['source']}")
        print(f"Page     : {passage['page']}")
        print(f"Distance : {passage['distance']:.4f}")
        print(f"Accès    : {passage['access']}")
        print(f"Texte    : {passage['text'][:500]}...")


if __name__ == "__main__":
    main()
