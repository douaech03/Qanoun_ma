# ⚖️ Qanoun.ma

**Assistant documentaire RAG dédié au droit marocain**

## 1. Présentation du projet

**Qanoun.ma** est une application universitaire qui permet de poser des questions en français sur des documents juridiques marocains. Elle s'appuie sur une architecture **RAG (Retrieval-Augmented Generation)** : une recherche documentaire retrouve des passages pertinents, puis un modèle de langage local formule une réponse à partir de ces passages et présente leurs références.

L'application est conçue pour un usage **pédagogique et documentaire**. Elle ne remplace pas un professionnel du droit.

| Informations | Détails |
|---|---|
| Établissement | ENSA Fès |
| Filière | ISDIA3 — Ingénierie en Science des Données et IA |
| Module | MLOps |
| Année universitaire | 2026–2027 |
| Encadrant | Mohammed AIRAJ |
| Projet | RAG multi-sources — reproductibilité, évaluation et sécurité documentaire |
| Équipe | Amiri Abdelhadi et Chaibi Douae |
| Application | **Qanoun.ma** |
| Dépôt | https://github.com/douaech03/Qanoun_ma |

## 2. Objectifs

- Exploiter des documents juridiques de différentes sources et conserver leurs métadonnées de provenance.
- Extraire et découper les textes pour construire **un index vectoriel ChromaDB unique**.
- Utiliser le modèle d'embeddings `intfloat/multilingual-e5-small` pour la recherche sémantique.
- Retrouver les passages pertinents tout en appliquant un filtrage par niveau d'accès.
- Générer les réponses en local avec **Ollama / `mistral:7b`** et afficher les sources disponibles.
- Tester la qualité des réponses, la fidélité aux documents et la sécurité documentaire.
- Favoriser la reproductibilité et le suivi des expériences par Git, DVC, MLflow et les paramètres de configuration.

Le **droit du travail marocain** constitue le premier domaine documentaire visé.

## 3. Fonctionnement général

```text
Documents juridiques (PDF, autres sources autorisées)
             |
             v
     Ingestion et extraction (src/ingest.py)
             |
             v
        Découpage en passages
             |
             v
        Embeddings multilingues E5
             |
             v
       Index unique ChromaDB
             |
             v
Question utilisateur --> Recherche (src/retriever.py)
                              |
                              v
                  Passages + métadonnées
                              |
                              v
                    Génération (src/rag.py)
                              |
                              v
                        Ollama / Mistral 7B
                              |
                              v
               Réponse et références (app.py)
```

**Répartition du travail :** Chaibi Douae développe l'ingestion, les embeddings, l'indexation et la recherche documentaire ; Amiri Abdelhadi développe la génération avec le LLM, les tests associés et l'interface Streamlit. L'intégration et la validation du système complet sont effectuées ensemble.

## 4. Technologies

| Technologie | Rôle |
|---|---|
| Python 3.11.9 | Développement |
| Streamlit | Interface de discussion |
| PyMuPDF | Extraction de texte depuis les PDF |
| Sentence Transformers / E5 | Embeddings multilingues |
| ChromaDB | Stockage et recherche vectorielle |
| Ollama / Mistral 7B | Génération locale des réponses |
| PyYAML | Chargement de `params.yaml` |
| pytest | Tests automatisés |
| DVC | Versionnement et reproductibilité des données et artefacts, selon configuration |
| MLflow | Suivi des expériences, selon configuration |
| psutil | Mesures de ressources pour le monitoring |
| Git / GitHub | Collaboration et versionnement du code |

## 5. Principaux fichiers

```text
Qanoun_ma/
├── app.py                  # Interface Streamlit
├── params.yaml             # Paramètres du pipeline
├── requirements.txt        # Dépendances Python
├── README.md               # Documentation du projet
├── src/
│   ├── ingest.py           # Préparation et indexation du corpus
│   ├── retriever.py        # Recherche documentaire
│   ├── rag.py              # Prompt et génération via Ollama
│   ├── monitoring.py       # Mesure et journalisation des requêtes
│   └── evaluate.py         # Fonctions d'évaluation
├── tests/
│   ├── test_ingest.py
│   └── test_rag.py
├── data/
│   ├── manifest.csv        # Inventaire des documents
│   └── evaluation/         # Questions de référence
├── docs/
│   ├── cadrage.md
│   └── reproductibilite.md
└── .github/workflows/ci.yml
```

L'arborescence peut évoluer avec l'ajout de nouveaux retrievers, scripts d'évaluation ou ressources graphiques.

## 6. Installation sous Windows (CMD)

**Prérequis :** Python 3.11, Git et [Ollama](https://ollama.com/).

```cmd
 git clone https://github.com/douaech03/Qanoun_ma.git
 cd Qanoun_ma
 python -m venv .venv
 .venv\Scripts\activate.bat
 python -m pip install -r requirements.txt
 ollama pull mistral:7b
```

Sur une installation où Ollama ne tourne pas automatiquement, démarrer son serveur avant d'utiliser l'application. Vérifier les modèles installés :

```cmd
ollama list
```

## 7. Configuration

Le fichier `params.yaml` centralise notamment :

```yaml
llm_model: mistral:7b
ollama_url: http://localhost:11434
temperature: 0.1
k: 4
collection: maroclex
index_dir: artifacts/index
model: intfloat/multilingual-e5-small
chunk_tokens: 400
overlap: 40
max_distance: null
```

`k` désigne le nombre maximal de passages recherchés. Avec `max_distance: null`, aucun seuil de distance n'est appliqué : **un passage retourné n'est donc pas automatiquement pertinent**.

## 8. Exécution et tests

**Lancer l'interface :**

```cmd
python -m streamlit run app.py
```

Ouvrir ensuite http://localhost:8501.

**Tester la génération avec un passage fictif :**

```cmd
python -m src.rag
```

**Tester le retriever, uniquement après mise à disposition du corpus et construction d'un index local :**

```cmd
python -m src.retriever "Quelle est la durée du congé annuel ?"
```

**Construire l'index, une fois les documents préparés et le manifeste complété :**

```cmd
python -m src.ingest
```

**Exécuter les tests :**

```cmd
python -m pytest tests/ -v
```

**Consulter le résumé du monitoring :**

```cmd
python -m src.monitoring
```

Le monitoring n'enregistre les métriques que pour les requêtes passant effectivement par `generate_with_monitoring()` ; un affichage de `0` est normal avant intégration de ce wrapper.

## 9. Données et traçabilité

Les documents juridiques et leurs métadonnées sont préparés par l'équipe d'ingestion. Le fichier `data/manifest.csv` permet de référencer le corpus. Les références affichées par Qanoun.ma doivent provenir des métadonnées des passages retrouvés, et non de références inventées par le modèle.

Le partage des PDF et de l'index ChromaDB doit être prévu séparément si ces éléments ne sont pas suivis par Git ; leur présence sur GitHub ne doit pas être supposée.

## 10. Sécurité et limites

- Le retriever applique par défaut un filtre d'accès aux documents **publics**.
- Le moteur de génération refuse d'appeler le LLM lorsqu'aucun passage exploitable n'est fourni.
- Les textes documentaires sont présentés au modèle comme des **données**, et non comme des instructions à suivre.
- Les journaux de monitoring sont conçus pour ne pas enregistrer le texte des questions et des documents.

Ces protections doivent être testées : **un prompt ne suffit pas à garantir** l'absence d'hallucinations, de réponses non fondées ou d'injections malveillantes. Les autorisations et les références doivent être validées au niveau de l'application.

## 11. État d'avancement

**Déjà vérifié :** le module `src/rag.py` communique avec Ollama / Mistral 7B et produit une réponse sourcée sur un **document synthétique de test** ; le script `src/monitoring.py` s'exécute et affiche un résumé initial.

**À valider sur le pipeline complet :** récupération des documents réels, construction ou partage de l'index, branchement du retriever à `app.py`, résultats des tests automatisés, fiabilité des références, sécurité des accès, suivi effectif des requêtes et expériences MLOps (DVC / MLflow).

Aucun résultat chiffré de qualité ou de sécurité n'est revendiqué sans exécution et mesure documentées.

---

**Qanoun.ma** — Projet MLOps, ENSA Fès, ISDIA3 — 2026–2027.
