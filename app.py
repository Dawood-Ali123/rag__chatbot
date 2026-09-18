import streamlit as st
import json
from pathlib import Path

from ingestion.ingest import ingest_document
from rag.crag import run_crag
from rag.self_rag import run_self_rag
import uuid

# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"
RESULTS_FILE = PROJECT_DIR / "evaluation" / "results.json"

if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages=[]
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG Strategy Platform",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 RAG Platform")

    st.caption("Intelligent RAG Strategy Comparison")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["💬 Chat", "📊 Evaluation"]
    )

    st.divider()

    st.subheader("🧠 RAG Strategies")

    st.info(
        "CRAG\n\n"
        "Corrective Retrieval-Augmented Generation"
    )

    st.info(
        "Self-RAG\n\n"
        "Self-Reflective Retrieval-Augmented Generation"
    )

    st.divider()

    st.subheader("⚙️ Technology")

    st.write("🔗 LangChain")
    st.write("🧠 LangGraph")
    st.write("📊 Ragas")
    st.write("🗂️ FAISS")
    st.write("🤖 Groq")


# =========================================================
# CHAT PAGE
# =========================================================

if page == "💬 Chat":

    st.title("🤖 RAG Strategy Platform")

    st.caption(
        "Upload your documents and compare intelligent RAG strategies."
    )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENT SECTION
    # -----------------------------------------------------

    st.header("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"✅ {len(uploaded_files)} document(s) ready"
        )

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        if st.button(
            "⚡ Process Documents",
            use_container_width=True
        ):

            UPLOAD_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            for file in uploaded_files:

                file_path = UPLOAD_DIR / file.name

                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            with st.spinner(
                "Building your knowledge base..."
            ):

                result = ingest_document()
            st.session_state.thread_id=str(uuid.uuid4())

            st.success(
                f"✅ Processed {result['documents']} document(s) "
                f"into {result['chunks']} chunks."
            )

    st.divider()

    # -----------------------------------------------------
    # STRATEGY SECTION
    # -----------------------------------------------------

    st.header("🧠 Choose RAG Strategy")

    strategy = st.radio(
        "Select the strategy you want to use:",
        ["CRAG", "Self-RAG"],
        horizontal=True
    )

    if strategy == "CRAG":

        st.info(
            "🔍 CRAG evaluates retrieved information and "
            "corrects poor retrieval when necessary."
        )

    else:

        st.info(
            "🔄 Self-RAG evaluates retrieval, support, "
            "and answer quality during generation."
        )

    st.divider()

    # -----------------------------------------------------
    # CHAT SECTION
    # -----------------------------------------------------

    st.header("💬 Chat with your Documents")

    st.caption(
        "Ask questions about the information contained in your PDFs."
    )

    for message in st.session_state.messages:
        with st.chat_message(
            message['role'],
            avatar="👤" if message["role"] == "user" else "🤖"
        ):
            st.write(message['content'])

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:
        st.session_state.messages.append({
            "role":"user",
            "content":question
        })
        st.chat_message(
            "user",
            avatar="👤"
        ).write(question)
        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            if not VECTORSTORE_DIR.exists():

                st.error(
                    "⚠️ Please upload and process documents first."
                )

            else:

                if strategy == "CRAG":

                    with st.spinner(
                        "CRAG is analyzing your question..."
                    ):

                        result = run_crag(question,
                                          st.session_state.thread_id)

                else:

                    with st.spinner(
                        "Self-RAG is analyzing your question..."
                    ):

                        result = run_self_rag(question,st.session_state.thread_id)

                answer = result.get(
                    "answer",
                    "No answer found."
                )

                st.write(answer)
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer
                })


# =========================================================
# EVALUATION PAGE
# =========================================================

else:

    st.title("📊 RAG Evaluation Dashboard")

    st.caption(
        "Evaluate and compare CRAG and Self-RAG using Ragas."
    )

    st.divider()

    # -----------------------------------------------------
    # CHECK RESULTS
    # -----------------------------------------------------

    if not RESULTS_FILE.exists():

        st.info(
            "📋 No evaluation results available yet."
        )

        st.write(
            "Run the Ragas evaluation first. "
            "The results will appear here automatically."
        )

    else:

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

        if not results:

            st.warning(
                "⚠️ Evaluation results are empty."
            )

        else:

            # =============================================
            # SUMMARY
            # =============================================

            st.header("📈 Overall Performance")

            metrics = [
                "Faithfulness",
                "Answer Relevancy",
                "Context Precision",
                "Context Recall"
            ]

            metric_keys = [
                "faithfulness",
                "relevancy",
                "precision",
                "recall"
            ]

            crag_scores = [
                sum(
                    r[f"crag_{key}"]
                    for r in results
                ) / len(results)
                for key in metric_keys
            ]

            self_scores = [
                sum(
                    r[f"self_rag_{key}"]
                    for r in results
                ) / len(results)
                for key in metric_keys
            ]

            # =============================================
            # METRIC CARDS
            # =============================================

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "🎯 Faithfulness",
                f"{crag_scores[0]:.2f}",
                f"Self-RAG {self_scores[0]:.2f}"
            )

            col2.metric(
                "💬 Answer Relevancy",
                f"{crag_scores[1]:.2f}",
                f"Self-RAG {self_scores[1]:.2f}"
            )

            col3.metric(
                "📚 Context Precision",
                f"{crag_scores[2]:.2f}",
                f"Self-RAG {self_scores[2]:.2f}"
            )

            col4.metric(
                "🔍 Context Recall",
                f"{crag_scores[3]:.2f}",
                f"Self-RAG {self_scores[3]:.2f}"
            )

            st.divider()

            # =============================================
            # COMPARISON TABLE
            # =============================================

            st.header("⚖️ CRAG vs Self-RAG")

            comparison_data = {
                "Metric": metrics,
                "CRAG": crag_scores,
                "Self-RAG": self_scores
            }

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )

            # =============================================
            # CHART
            # =============================================

            st.header("📊 Metric Comparison")

            chart_data = {
                "CRAG": crag_scores,
                "Self-RAG": self_scores
            }

            st.bar_chart(chart_data)

            st.divider()

            # =============================================
            # QUESTION RESULTS
            # =============================================

            st.header("📋 Question-wise Evaluation")

            question_data = []

            for result in results:

                question_data.append({

                    "ID": result["id"],

                    "Question": result["question"],

                    "CRAG Faithfulness": result.get(
                        "crag_faithfulness"
                    ),

                    "Self-RAG Faithfulness": result.get(
                        "self_rag_faithfulness"
                    ),

                    "CRAG Relevancy": result.get(
                        "crag_relevancy"
                    ),

                    "Self-RAG Relevancy": result.get(
                        "self_rag_relevancy"
                    ),

                    "CRAG Precision": result.get(
                        "crag_precision"
                    ),

                    "Self-RAG Precision": result.get(
                        "self_rag_precision"
                    ),

                    "CRAG Recall": result.get(
                        "crag_recall"
                    ),

                    "Self-RAG Recall": result.get(
                        "self_rag_recall"
                    )
                })

            st.dataframe(
                question_data,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Total evaluated questions: {len(results)}"
            )