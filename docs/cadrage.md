# Cadrage du projet — Qanoun.ma

## 1. Identification du projet

- **Application :** Qanoun.ma
- **Établissement :** ENSA Fès
- **Filière :** ISDIA3 — Ingénierie en Science des Données et Intelligence Artificielle
- **Module :** MLOps
- **Année universitaire :** 2026–2027
- **Encadrement :** Mohammed AIRAJ
- **Équipe :** Amiri Abdelhadi et Chaibi Douae
- **Sujet :** RAG multi-sources — reproductibilité, évaluation et sécurité documentaire

## 2. Contexte et problématique

Les textes juridiques peuvent être longs, répartis entre plusieurs documents et difficiles à consulter rapidement. Un modèle de langage utilisé seul peut produire une réponse plausible sans preuve documentaire fiable.

**Problématique :** comment permettre à un utilisateur de poser une question en français sur le droit du travail marocain et d'obtenir une réponse fondée sur des passages documentaires identifiables, tout en assurant la traçabilité, la reproductibilité et un contrôle d'accès aux sources ?

Qanoun.ma est un **assistant documentaire pédagogique** : il ne remplace pas un avocat et ne délivre pas de conseil juridique individualisé.

## 3. Objectif général

Développer une application RAG locale qui recherche des passages dans un corpus juridique multi-sources, les transmet à un LLM et présente une réponse accompagnée des références disponibles.

### Objectifs spécifiques

1. Préparer des documents de plusieurs formats avec leurs métadonnées de provenance.
2. Construire **un index vectoriel unique** pour les documents autorisés à l'indexation.
3. Rechercher les passages pertinents en filtrant les niveaux d'accès.
4. Générer les réponses avec **Mistral 7B via Ollama**, à partir des passages fournis.
5. Afficher les références des documents et les pages lorsqu'elles sont connues.
6. Tester le refus lorsque le contexte ne permet pas de répondre, les erreurs et les tentatives d'injection d'instructions.
7. Comparer des configurations de recherche ou de découpage au moyen de mesures documentées.
8. Suivre les paramètres, les versions, les tests et les métriques dans une démarche MLOps.

## 4. Périmètre du projet

### Inclus — version initiale

- Domaine principal : **droit du travail marocain**.
- Questions en français sur les sujets couverts par les documents réellement présents dans le corpus.
- Ingestion des PDF et, selon le corpus retenu, de documents textuels/Markdown.
- Découpage en passages, embeddings multilingues E5, recherche avec ChromaDB.
- Métadonnées de provenance (source, identifiant documentaire, page si disponible, niveau d'accès, version du corpus).
- Réponses locales avec Mistral 7B et interface conversationnelle Streamlit.
- Tests automatisés, évaluation de la recherche, sécurité documentaire et monitoring de base.

### Hors périmètre — version initiale

- Consultation juridique personnalisée ou garantie d'exactitude juridique.
- Recherche automatique sur l'ensemble du Web en temps réel.
- Garantie que les documents reflètent immédiatement toutes les évolutions législatives.
- Déploiement cloud, comptes utilisateurs complets ou base de données persistante d'historique des conversations.
- Extension à tous les domaines du droit marocain dès la première version.

## 5. Corpus documentaire et données

Les données exploitées sont **principalement textuelles**. Les PDF sont des conteneurs à partir desquels le texte est extrait avant indexation.

| Type de document | Rôle | État à vérifier |
|---|---|---|
| Code du travail marocain, loi n° 65-99 (PDF) | Source juridique principale prévue | Fichier, version et date à confirmer dans `data/manifest.csv` |
| Autres documents institutionnels/explicatifs (PDF) | Source complémentaire prévue | Disponibilité et fiabilité à contrôler |
| FAQ pédagogique (Markdown, si retenue) | Questions/réponses rattachées à des sources vérifiées | À préparer et sourcer |
| `data/manifest.csv` | Inventaire, provenance et niveaux d'accès | À compléter/contrôler avec les documents réels |
| Documents fictifs de test | Tester les refus, permissions et injections | Ne jamais présenter comme du droit réel |

Les documents bruts sont destinés à `data/raw/`. Les fichiers du corpus et les index peuvent nécessiter DVC ou une procédure de reconstruction plutôt qu'un simple `git pull`.

**Règle importante :** un document non vérifié, absent, obsolète ou fictif ne doit pas être présenté comme une source juridique officielle.

## 6. Fonctionnement du système

```text
Documents + manifest.csv
       |
       v
Ingestion et extraction du texte
       |
       v
Découpage en passages + métadonnées
       |
       v
Embeddings E5 -> index ChromaDB unique
                       |
Question utilisateur -> Retriever (filtre d'accès, top-k)
                       |
                       v
           Passages et références
                       |
                       v
         Prompt contrôlé (src/rag.py)
                       |
                       v
        Ollama / Mistral 7B local
                       |
                       v
       Réponse + sources dans Streamlit
```

Le retriever fournit des dictionnaires comprenant notamment `text`, `source`, `page`, `distance` et `access`. Le moteur `generate_answer(question, passages)` exploite ces résultats sans refaire l'indexation.

Les paramètres sont centralisés dans `params.yaml` (modèle, révision, taille des passages, recouvrement, `k`, collection et modèle LLM). Le seuil `max_distance` n'est pas encore défini : un résultat de recherche n'est donc pas automatiquement une preuve de pertinence suffisante.

## 7. Répartition du travail

| Responsabilité | Personne |
|---|---|
| Préparation et inventaire du corpus | Chaibi Douae |
| Ingestion, découpage, embeddings et indexation | Chaibi Douae |
| Recherche documentaire et filtrage initial | Chaibi Douae |
| Génération des réponses avec Ollama/Mistral | Amiri Abdelhadi |
| Interface Streamlit et tests du module de génération | Amiri Abdelhadi |
| Intégration finale, évaluation, sécurité et rapport | Travail commun |

Les contributions sont développées sur des branches Git puis intégrées par Pull Request.

## 8. Exigences fonctionnelles

- **EF1 :** l'utilisateur peut saisir une question dans l'interface.
- **EF2 :** le système interroge le corpus autorisé et retourne jusqu'à `k` passages pertinents.
- **EF3 :** le LLM reçoit la question et les passages documentaires, avec leurs identifiants de citation.
- **EF4 :** l'application affiche la réponse et les métadonnées disponibles des sources.
- **EF5 :** si aucun passage exploitable n'est disponible, l'application affiche un refus explicite.
- **EF6 :** les documents fictifs sont signalés comme tels en démonstration.
- **EF7 :** les documents non autorisés ne doivent pas parvenir au LLM.

## 9. Exigences non fonctionnelles et MLOps

- **Traçabilité :** conserver les identifiants et métadonnées de provenance.
- **Reproductibilité :** enregistrer les configurations, versions des documents et du modèle d'embeddings ; documenter la reconstruction de l'index.
- **Qualité :** exécuter des tests unitaires et une évaluation sur des questions annotées ; mesurer notamment **Recall@k** et **MRR** pour le retrieval et contrôler manuellement ou selon un protocole explicite la fidélité des réponses/citations.
- **Sécurité :** filtrer les accès avant génération ; traiter les textes des documents comme des données et évaluer la résistance aux injections de prompt.
- **Confidentialité :** éviter de journaliser les questions et les textes sensibles ; surveiller principalement des métriques techniques.
- **Observabilité :** enregistrer au minimum le statut, les erreurs et le temps de réponse des appels effectivement instrumentés.
- **Qualité du déploiement :** utiliser des tests pytest et une CI exécutable sans API LLM payante pour les vérifications de base.

Les seuils de performance ne seront fixés qu'après définition d'une baseline ou d'un critère justifié ; aucun résultat chiffré n'est présumé acquis.

## 10. Technologies retenues

- **Python 3.11.9** : développement.
- **PyMuPDF** : extraction de texte PDF.
- **Sentence Transformers — multilingual E5** : embeddings.
- **ChromaDB** : index vectoriel.
- **Ollama + `mistral:7b`** : génération locale.
- **Streamlit** : interface.
- **pytest** : tests.
- **Git/GitHub** : collaboration et versionnement du code.
- **DVC et MLflow** : reproductibilité et suivi des expériences, selon les pipelines mis en place.
- **psutil / JSONL** : métriques techniques de monitoring.

## 11. Risques et limites

| Risque | Mesure prévue |
|---|---|
| Sources absentes ou versions anciennes | Vérifier le corpus et documenter dates/versions |
| Passages hors sujet malgré la recherche | Évaluer la pertinence ; calibrer un éventuel seuil |
| Hallucinations ou références inventées | Prompt restrictif, refus, tests de fidélité et vérification des citations |
| Instructions malveillantes dans un document | Séparer données/instructions et tester des attaques concrètes |
| Fuite de documents restreints | Contrôle d'accès avant récupération et avant génération |
| Dépendance au matériel local | Tester Ollama sur le poste cible et documenter les prérequis |
| Différences entre postes | Conserver paramètres, dépendances et procédure de reconstruction |

## 12. Livrables attendus

1. Dépôt GitHub **Qanoun_ma** avec le code source et les tests.
2. Application **Qanoun.ma** utilisable en démonstration.
3. Corpus documenté avec `data/manifest.csv` et méthode de reconstruction de l'index.
4. Résultats d'évaluation réellement mesurés et enregistrés.
5. Preuves de tests, contrôles de sécurité et procédure de monitoring.
6. `README.md`, documentation de cadrage/reproductibilité, rapport et support de soutenance.

## 13. État actuel et prochaines validations

**Développé :** le module `src/rag.py` a été exécuté avec Ollama et un passage entièrement fictif ; une interface Streamlit et les fichiers de recherche documentaire existent ; des tests automatisés ont été préparés. Le module de monitoring existe, mais sa collecte n'est pas encore reliée à toutes les requêtes de l'application.

**À vérifier avant de conclure au fonctionnement complet :** disponibilité des PDF et de l'index ChromaDB sur chaque PC, connexion de `src/retriever.py` à `app.py`, résultats pytest, contrôle des permissions, fidélité des citations, expériences MLflow/DVC et mesures d'évaluation.

---

*Document de cadrage académique — Qanoun.ma — année universitaire 2026–2027.*
