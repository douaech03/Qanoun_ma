
"""Recherche hybride E5 + BM25 + RRF pour MarocLex AI."""

import argparse
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# 1. Configuration et connexion à ChromaDB
# ==================================================

def load_config():
    """Charger la configuration principale du projet."""

    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=1)
def load_model():
    """Charger E5 une seule fois pendant l'exécution."""

    config = load_config()

    return SentenceTransformer(
        config["model"],
        revision=str(config["revision"]),
    )


def get_collection():
    """Ouvrir l'index ChromaDB principal sans le modifier."""

    config = load_config()
    index_dir = ROOT / config["index_dir"]

    if not index_dir.is_dir():
        raise FileNotFoundError(
            f"Index introuvable : {index_dir}"
        )

    client = chromadb.PersistentClient(path=str(index_dir))

    return client.get_collection(
        name=config["collection"]
    )


# ==================================================
# 2. Tokenisation pour BM25
# ==================================================

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de",
    "d", "l", "au", "aux", "en", "et", "ou", "a",
    "à", "est", "sont", "être", "pour", "par", "sur",
    "dans", "avec", "qui", "que", "qu", "quoi",
    "quel", "quelle", "quelles", "quels",
    "ce", "ces", "se", "son", "sa", "ses",
    "il", "elle", "ils", "elles", "ne", "pas",
    "comme", "entre", "concernant", "prevues",
    "prevu", "cas", "existe", "t", "on",
}


def tokenize(text):
    """Normaliser le texte et extraire ses mots utiles."""

    # Convertir les lettres accentuées
    text = unicodedata.normalize("NFKD", text.lower())

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Garder les mots et les nombres
    words = re.findall(r"[a-z0-9]+", text)

    tokens = []

    for word in words:
        if len(word) <= 1 or word in STOPWORDS:
            continue

        # Normalisation légère des pluriels
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]

        tokens.append(word)

    return tokens


# ==================================================
# 3. Charger les chunks publics pour BM25
# ==================================================

def load_public_chunks(collection):
    """Lire uniquement les chunks publics de ChromaDB."""

    data = collection.get(
        where={"access": "public"},
        include=["documents", "metadatas"],
    )

    chunks = []

    for chunk_id, text, metadata in zip(
        data["ids"],
        data["documents"],
        data["metadatas"],
    ):
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata,
        })

    return chunks


# ==================================================
# 4. Calculer les scores BM25
# ==================================================

def bm25_search(question, chunks, limit=100):
    """Classer les chunks selon les mots-clés de la question."""

    if not chunks:
        return []

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    documents_tokens = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    document_count = len(documents_tokens)

    avg_length = sum(
        len(tokens) for tokens in documents_tokens
    ) / document_count

    if avg_length == 0:
        return []

    # Fréquence documentaire de chaque mot
    document_frequency = Counter()

    for tokens in documents_tokens:
        document_frequency.update(set(tokens))

    k1 = 1.5
    b = 0.75

    scores = []

    for chunk, tokens in zip(chunks, documents_tokens):
        term_frequency = Counter(tokens)
        doc_length = len(tokens)
        score = 0.0

        for term in set(query_tokens):
            tf = term_frequency.get(term, 0)

            if tf == 0:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (document_count - df + 0.5) / (df + 0.5)
            )

            numerator = tf * (k1 + 1)

            denominator = tf + k1 * (
                1 - b + b * doc_length / avg_length
            )

            score += idf * numerator / denominator

        # Ignorer les chunks sans correspondance lexicale
        if score > 0:
            scores.append((chunk["id"], score))

    scores.sort(
        key=lambda item: (-item[1], item[0])
    )

    return scores[:limit]


# ==================================================
# 5. Recherche sémantique E5
# ==================================================

def semantic_search(question, collection, limit=100):
    """Récupérer les chunks par similarité vectorielle."""

    config = load_config()
    model = load_model()

    vector = model.encode(
        ["query: " + question],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(limit, collection.count()),
        where={"access": "public"},
        include=["distances"],
    )

    max_distance = config.get("max_distance")
    ranked = []

    for chunk_id, distance in zip(
        results["ids"][0],
        results["distances"][0],
    ):
        if (
            max_distance is not None
            and distance > float(max_distance)
        ):
            continue

        ranked.append((chunk_id, float(distance)))

    return ranked


# ==================================================
# 6. Fusion RRF
# ==================================================

def reciprocal_rank_fusion(
    semantic_results,
    lexical_results,
    rrf_k=60,
):
    """Fusionner les classements E5 et BM25."""

    combined_scores = {}

    for rank, (chunk_id, _) in enumerate(
        semantic_results, start=1
    ):
        combined_scores[chunk_id] = (
            combined_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    for rank, (chunk_id, _) in enumerate(
        lexical_results, start=1
    ):
        combined_scores[chunk_id] = (
            combined_scores.get(chunk_id, 0.0)
            + 1.0 / (rrf_k + rank)
        )

    return sorted(
        combined_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )


# ==================================================
# 7. Retriever hybride complet
# ==================================================

def hybrid_retrieve(
    question,
    k=4,
    semantic_pool=100,
    lexical_pool=100,
):
    """Rechercher et retourner les k meilleurs chunks publics."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("La question ne doit pas être vide.")

    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        raise ValueError("k doit être un entier positif.")

    if (
        not isinstance(semantic_pool, int)
        or isinstance(semantic_pool, bool)
        or semantic_pool <= 0
        or not isinstance(lexical_pool, int)
        or isinstance(lexical_pool, bool)
        or lexical_pool <= 0
    ):
        raise ValueError("Les tailles des pools doivent être positives.")

    collection = get_collection()

    # Tous les documents utilisés ici sont publics
    chunks = load_public_chunks(collection)

    if not chunks:
        return []

    # Recherche vectorielle
    semantic_results = semantic_search(
        question,
        collection,
        limit=semantic_pool,
    )

    # Recherche par mots-clés
    lexical_results = bm25_search(
        question,
        chunks,
        limit=lexical_pool,
    )

    # Fusion des deux classements
    fused = reciprocal_rank_fusion(
        semantic_results,
        lexical_results,
    )

    # Retrouver le texte et les métadonnées
    chunks_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }

    distances = dict(semantic_results)

    output = []

    for chunk_id, rrf_score in fused[:k]:
        chunk = chunks_by_id[chunk_id]
        metadata = chunk["metadata"]

        output.append({
            "id": chunk_id,
            "text": chunk["text"],
            "document_id": metadata["document_id"],
            "source": metadata["source"],
            "page": metadata["page"],
            "access": metadata["access"],
            "file": metadata.get("file"),
            "distance": distances.get(chunk_id),
            "rrf_score": rrf_score,
        })

    return output


# ==================================================
# 8. Exécution depuis PowerShell
# ==================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recherche hybride MarocLex AI"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question juridique",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Nombre de passages retournés",
    )

    args = parser.parse_args()

    results = hybrid_retrieve(
        args.question,
        k=args.k,
    )

    print(f"\nNombre de passages trouvés : {len(results)}")

    for rank, item in enumerate(results, start=1):
        print(f"\n--- Passage {rank} ---")
        print(f"Source    : {item['source']}")
        print(f"Document  : {item['document_id']}")
        print(f"Page      : {item['page']}")
        print(f"Score RRF : {item['rrf_score']:.6f}")
        print(f"Texte     : {item['text'][:350]}...")


if __name__ == "__main__":
    main()
