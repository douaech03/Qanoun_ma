
"""
MarocLex AI - Interface utilisateur Streamlit
Responsable : Amiri Abdelhadi
"""

import streamlit as st

from src.rag import generate_answer


# ==========================================
# 1. Configuration de la page
# ==========================================

st.set_page_config(
    page_title="MarocLex AI",
    page_icon="⚖️",
    layout="centered",
)


# ==========================================
# 2. Fonction de récupération des passages
# ==========================================

def get_passages(question, demo_mode=True):
    """
    Fournir les passages au moteur RAG.

    Pour le moment :
    - Mode démonstration : document fictif.
    - Autre mode : aucun document disponible.

    Plus tard, cette fonction utilisera le
    retriever ChromaDB développé par Douae.
    """

    if demo_mode:
        return [
            {
                "text": (
                    "DOCUMENT ENTIEREMENT FICTIF. "
                    "Dans l'entreprise imaginaire Atlas-B, "
                    "le règlement de démonstration prévoit "
                    "un délai de réponse de trois jours "
                    "ouvrables pour les demandes internes."
                ),
                "source": "DOCUMENT_FICTIF_TEST",
                "page": 1,
                "distance": None,
                "access": "public",
            }
        ]

    # Aucun retriever réel connecté actuellement
    return []


# ==========================================
# 3. En-tête de l'application
# ==========================================

st.title("⚖️ MarocLex AI")

st.write(
    "Assistant documentaire intelligent basé "
    "sur le RAG et le modèle local Mistral 7B."
)

st.caption(
    "Projet universitaire MLOps - ENSA Fès - ISDIA3"
)

st.divider()


# ==========================================
# 4. Mode de fonctionnement
# ==========================================

st.subheader("Poser une question")

st.warning(
    "Version de démonstration : la recherche "
    "documentaire réelle n'est pas encore connectée. "
    "Les exemples utilisés sont entièrement fictifs "
    "et ne constituent pas des références juridiques."
)


# ==========================================
# 5. Formulaire utilisateur
# ==========================================

with st.form("question_form"):

    question = st.text_area(
        "Votre question :",
        value=(
            "Quel est le délai de réponse prévu "
            "dans le document fictif ?"
        ),
        height=100,
    )

    demo_mode = st.checkbox(
        "Utiliser le document fictif de démonstration",
        value=True,
    )

    submitted = st.form_submit_button(
        "Générer la réponse",
        type="primary",
        use_container_width=True,
    )


# ==========================================
# 6. Appel au moteur RAG
# ==========================================

if submitted:

    st.session_state["last_demo_mode"] = demo_mode

    if not question.strip():
        st.session_state["last_result"] = {
            "status": "invalid_question",
            "answer": "Veuillez saisir une question.",
            "sources": [],
        }

    else:
        passages = get_passages(
            question,
            demo_mode=demo_mode,
        )

        with st.spinner("Génération en cours..."):
            try:
                result = generate_answer(
                    question,
                    passages,
                )

            except Exception:
                result = {
                    "status": "error",
                    "answer": (
                        "Une erreur technique est survenue. "
                        "Vérifiez Ollama et la configuration."
                    ),
                    "sources": [],
                }

        st.session_state["last_result"] = result


# ==========================================
# 7. Affichage des résultats
# ==========================================

if "last_result" in st.session_state:

    result = st.session_state["last_result"]

    st.divider()
    st.subheader("Réponse de MarocLex AI")

    if result["status"] == "ok":

        if st.session_state.get("last_demo_mode"):
            st.info(
                "Réponse basée uniquement sur "
                "un document synthétique de test."
            )

        st.markdown(result["answer"])

    elif result["status"] == "insufficient_context":
        st.warning(result["answer"])

    else:
        st.error(result["answer"])

    # Afficher les sources documentaires
    if result["sources"]:

        st.subheader("Sources documentaires fournies")

        for source in result["sources"]:

            reference = source["reference"]
            document = source["source"]
            page = source["page"]
            distance = source["distance"]

            with st.expander(
                f"[{reference}] {document}"
            ):

                st.write(f"Document : {document}")

                if page is not None:
                    st.write(f"Page : {page}")
                else:
                    st.write("Page : non disponible")

                if distance is not None:
                    st.write(
                        f"Distance de recherche : {distance}"
                    )

        st.caption(
            "Ces références correspondent aux passages "
            "transmis au modèle. Leur utilisation effective "
            "dans la réponse reste à vérifier."
        )


# ==========================================
# 8. Pied de page
# ==========================================

st.divider()

st.caption(
    "MarocLex AI - Projet MLOps 2026-2027 | "
    "Chaibi Douae & Amiri Abdelhadi"
)
