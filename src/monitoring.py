
"""
MarocLex AI - Monitoring du système RAG
Module MLOps - ENSA Fès - ISDIA3
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from src.rag import generate_answer


ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "artifacts" / "monitoring" / "requests.jsonl"


def log_request(status, latency, sources_count=0, error_type=None):
    """Enregistrer les métriques d'une requête RAG."""

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "latency_seconds": round(latency, 3),
        "sources_count": sources_count,
        "memory_mb": round(
            process.memory_info().rss / (1024 * 1024), 2
        ),
        "error_type": error_type,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")

    return event


def generate_with_monitoring(question, passages):
    """
    Générer une réponse RAG et mesurer ses performances.

    Ne conserve ni la question ni le texte des documents.
    """

    start_time = time.perf_counter()

    try:
        result = generate_answer(question, passages)

        latency = time.perf_counter() - start_time

        log_request(
            status=result["status"],
            latency=latency,
            sources_count=len(result.get("sources", [])),
        )

        return result

    except Exception as error:
        latency = time.perf_counter() - start_time

        log_request(
            status="error",
            latency=latency,
            error_type=type(error).__name__,
        )

        raise


def monitoring_summary():
    """Calculer un résumé des requêtes enregistrées."""

    if not LOG_FILE.exists():
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "average_latency": 0,
        }

    events = []

    with open(LOG_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))

    if not events:
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "average_latency": 0,
        }

    total = len(events)
    successful = sum(
        event["status"] == "ok" for event in events
    )

    average_latency = sum(
        event["latency_seconds"] for event in events
    ) / total

    return {
        "total_requests": total,
        "successful_requests": successful,
        "average_latency": round(average_latency, 3),
    }


if __name__ == "__main__":
    print("=== MONITORING MAROCLEX AI ===")

    summary = monitoring_summary()

    print("Requêtes totales :", summary["total_requests"])
    print("Requêtes réussies :", summary["successful_requests"])
    print("Latence moyenne :", summary["average_latency"], "s")
