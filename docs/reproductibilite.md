
# Reproductibilité du projet Qanoun.ma

## 1. Présentation

**Qanoun.ma** est un assistant juridique basé sur une architecture RAG (Retrieval-Augmented Generation), développé dans le cadre du module MLOps à l'ENSA Fès.

Réalisé par :
- Amiri Abdelhadi
- Chaibi Douae

**Objectif :** permettre à un autre étudiant de reconstruire et de tester le système avec les mêmes documents, paramètres et versions logicielles.

La reproductibilité concerne :
- Les données documentaires.
- Les paramètres d'ingestion.
- Le modèle d'embeddings.
- L'index vectoriel ChromaDB.
- La recherche documentaire.
- Le modèle de génération.
- Les tests et les résultats d'évaluation.

---

## 2. Environnement de développement

| Élément | Technologie |
|---|---|
| Langage | Python 3.11.9 |
| Interface | Streamlit |
| Extraction PDF | PyMuPDF |
| Embeddings | Sentence Transformers |
| Modèle d'embeddings | multilingual-e5-small |
| Index vectoriel | ChromaDB |
| LLM | Mistral 7B |
| Exécution du LLM | Ollama |
| Tests | pytest |
| Versionnement | Git et GitHub |
| Suivi des données | DVC (prévu) |
| Suivi des expériences | MLflow (prévu) |

### Installation

Cloner le dépôt :

```bash
git clone https://github.com/douaech03/Qanoun_ma.git
cd Qanoun_ma
```

Créer et activer un environnement virtuel sous Windows :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

Vérifier Python :

```powershell
python --version
```

Pour enregistrer les versions réellement utilisées :

```powershell
python -m pip freeze > versions_installees.txt
```

Le fichier `requirements.txt` contient des plages de versions. Il ne garantit donc pas, à lui seul, une installation strictement identique.

---

## 3. Versionnement du code avec Git

Le code source est hébergé sur GitHub.

Dépôt :
https://github.com/douaech03/Qanoun_ma

Pour récupérer les dernières modifications :

```powershell
git pull --ff-only origin main
```

Pour identifier exactement la version du code :

```powershell
git rev-parse HEAD
```

Le hash du commit permet de retrouver la version du projet utilisée pendant une expérience.

Pour reproduire une expérience, il faut conserver :
- Le hash du commit Git.
- Les versions des dépendances.
- Les paramètres du pipeline.
- La version du corpus documentaire.
- Les résultats d'évaluation.

---

## 4. Reproductibilité des données

Les documents juridiques utilisés constituent le corpus documentaire de Qanoun.ma.

Exemples de sources :
- Code du travail marocain.
- Lois relatives au travail et à l'emploi.
- Documents juridiques publics pertinents.

Les documents sont préparés par Chaibi Douae.

### Organisation prévue

```text
data/
|-- raw/
|   |-- documents_juridiques.pdf
|
|-- manifest.csv
|
|-- evaluation/
|   |-- questions.csv
```

Le fichier `data/manifest.csv` sert à décrire les documents et leurs métadonnées.

Le corpus doit être identifiable par une version.

### Pourquoi versionner les données ?

Si un document est ajouté, modifié ou supprimé, les résultats de recherche peuvent changer.

Pour reproduire les mêmes résultats, il faut donc utiliser le même corpus documentaire.

Le partage et le versionnement des PDF avec DVC restent à mettre en place ou à valider.

---

## 5. Reproductibilité des paramètres

Les paramètres du pipeline sont centralisés dans `params.yaml`.

Exemples :

```yaml
embedding: e5
model: intfloat/multilingual-e5-small
revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3

chunk_tokens: 400
overlap: 40

collection: maroclex
index_dir: artifacts/index

k: 4
max_distance: null

llm_model: mistral:7b
ollama_url: http://localhost:11434
temperature: 0.1
```

### Rôle des paramètres

- `model` : modèle utilisé pour transformer les textes en vecteurs.
- `revision` : version précise du modèle d'embeddings.
- `chunk_tokens` : taille des passages documentaires.
- `overlap` : chevauchement entre deux passages.
- `collection` : nom technique de la collection ChromaDB.
- `index_dir` : emplacement de l'index vectoriel.
- `k` : nombre maximal de passages recherchés.
- `max_distance` : seuil optionnel de filtrage des résultats.
- `llm_model` : modèle utilisé pour générer la réponse.
- `temperature` : paramètre influençant la variabilité de la génération.

Les paramètres doivent être enregistrés pour chaque expérience.

Le nom technique `maroclex` de la collection ne change pas le nom de l'application, qui est **Qanoun.ma**.

---

## 6. Reconstruction de l'index vectoriel

L'index ChromaDB contient les embeddings des passages documentaires.

Pour reconstruire l'index, il faut :

1. Disposer du corpus documentaire attendu.
2. Vérifier le fichier `data/manifest.csv`.
3. Utiliser la configuration `params.yaml`.
4. Exécuter le pipeline d'ingestion.

Commande :

```powershell
python -m src.ingest
```

L'index est enregistré dans le répertoire configuré :

```text
artifacts/index/
```

### Condition importante

La reconstruction n'est possible que si les documents nécessaires sont disponibles localement et que leurs références dans le manifeste sont correctes.

Deux index construits à partir de corpus différents ne constituent pas une expérience identique.

---

## 7. Reproductibilité de la recherche documentaire

Le fichier `src/retriever.py` effectue une recherche sémantique dans ChromaDB.

La recherche suit ces étapes :

1. Réception de la question.
2. Transformation en embedding avec E5.
3. Filtrage des documents autorisés.
4. Recherche des `k` passages les plus proches.
5. Retour des textes et des métadonnées.

Exemple de test :

```powershell
python -m src.retriever "Quelle est la durée du congé annuel ?"
```

Chaque passage retrouvé contient notamment :
- Son identifiant.
- Son texte.
- Sa source.
- Sa page.
- Sa distance vectorielle.
- Son niveau d'accès.
- Sa version de corpus.

Pour reproduire la recherche, il faut conserver le même index, le même modèle E5 et les mêmes paramètres.

---

## 8. Reproductibilité de la génération

La génération est assurée par `src/rag.py`.

Le modèle utilisé est Mistral 7B, exécuté localement avec Ollama.

### Installation du modèle

```powershell
ollama pull mistral:7b
```

Vérification :

```powershell
ollama list
```

### Test du module RAG

```powershell
python -m src.rag
```

Le test autonome du module utilise un document fictif.

Dans l'application complète, les passages devront provenir du retriever.

### Limite

Même avec une température faible, les réponses d'un LLM ne sont pas garanties strictement identiques.

Pour une reproduction plus précise, il faut également enregistrer la version exacte du modèle Ollama et les conditions d'exécution.

---

## 9. Évaluation du système

L'évaluation vise à mesurer la qualité de la recherche et des réponses générées.

Les expériences peuvent comparer :
- Plusieurs tailles de chunks.
- Différentes valeurs de `k`.
- La recherche sémantique.
- La recherche hybride.
- La recherche multi-query.

Les indicateurs pertinents incluent :
- Recall@k pour la recherche.
- MRR pour le classement des passages.
- Latence des requêtes.
- Qualité des références.
- Fidélité des réponses aux documents.

Les métriques doivent être calculées sur un jeu de questions de référence.

Les scores ne doivent pas être présentés comme validés avant l'exécution effective des expériences.

---

## 10. Monitoring

Le fichier `src/monitoring.py` permet de collecter :
- Le nombre de requêtes.
- Le nombre de requêtes réussies.
- Le temps de réponse.
- Le nombre de sources retournées.
- La mémoire utilisée.
- Le type des erreurs rencontrées.

Commande :

```powershell
python -m src.monitoring
```

Les métriques sont enregistrées dans :

```text
artifacts/monitoring/requests.jsonl
```

Le monitoring doit être connecté au parcours réel de génération pour collecter automatiquement les requêtes de l'application.

Les questions et les passages documentaires ne sont pas enregistrés par ce module.

---

## 11. Tests automatiques

Les tests sont placés dans le dossier `tests/`.

Commande :

```powershell
python -m pytest tests/ -v
```

Ils servent à vérifier :
- L'ingestion.
- La recherche documentaire.
- La génération.
- La gestion des erreurs.
- Les règles de sécurité.

Les résultats réels de ces tests doivent être conservés lors des validations finales.

---

## 12. Sécurité et contrôle des accès

Qanoun.ma distingue notamment deux niveaux d'accès :

- `public` : document accessible à tous.
- `restricted` : document réservé à des utilisateurs autorisés.

Par défaut, le retriever limite la recherche aux documents publics.

Les instructions présentes dans les documents ne doivent pas être exécutées comme des instructions du système.

Pour reproduire une expérience de sécurité, il faut conserver les mêmes documents, les mêmes règles d'autorisation et les mêmes cas de test.

---

## 13. Limites actuelles

Les points suivants doivent encore être vérifiés ou finalisés :

- Constitution du corpus juridique réel.
- Partage reproductible des documents et de l'index.
- Connexion complète du retriever à l'interface Streamlit.
- Validation des réponses avec des documents réels.
- Suivi effectif des expériences dans MLflow.
- Versionnement des données avec DVC.
- Intégration automatique du monitoring.
- Exécution finale de tous les tests.

Ces éléments seront documentés après leur validation.

---

## 14. Conclusion

La reproductibilité est un objectif central du projet Qanoun.ma.

Elle repose sur la conservation du code, du corpus, des paramètres, des versions des modèles et des résultats d'évaluation.

Cette démarche permet de comparer les expériences et de reconstruire le pipeline RAG dans un environnement maîtrisé.
