from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

VECTORSTORE_DIR = "data/vectorstore"
def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

class State(TypedDict, total=False):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]

    evidence: List[str]
    retries: int

    isuse: Literal[
        "useful",
        "not_useful"
    ]

    use_reason: str


# =========================================================
# 4. Decide Whether Retrieval Is Needed
# =========================================================

class RetrieveDecision(BaseModel):

    should_retrieve: bool = Field(
        ...,
        description="True if company documents are needed."
    )


decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You decide whether document retrieval is needed.\n"

            "Return whether retrieval is required.\n\n"

            "Rules:\n"

            "- True if the question needs specific information "
            "from company documents.\n"

            "- False if the question is a general explanation "
            "or definition.\n"

            "- If unsure, choose True."
        ),

        (
            "human",
            "Question:\n{question}"
        )
    ]
)


decide_retrieval_chain = (
    decide_retrieval_prompt
    | llm.with_structured_output(RetrieveDecision)
)


def decide_retrieval(state: State):

    decision = decide_retrieval_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "need_retrieval": decision.should_retrieve
    }


def route_after_decide(state: State):

    if state["need_retrieval"]:
        return "retrieve"

    return "generate_direct"


# =========================================================
# 5. Direct Generation
# =========================================================

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "Answer using only your general knowledge.\n"

            "If the question requires specific company "
            "information, say:\n"

            "I don't know based on my general knowledge."
        ),

        (
            "human",
            "{question}"
        )
    ]
)


direct_generation_chain = (
    direct_generation_prompt
    | llm
)


def generate_direct(state: State):

    result = direct_generation_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "answer": result.content
    }


# =========================================================
# 6. Retrieve Documents
# =========================================================

def retrieve(state: State):

    query = (
        state.get("retrieval_query")
        or state["question"]
    )

    retriever = get_retriever()

    docs = retriever.invoke(query)

    return {
        "docs": docs
    }


# =========================================================
# 7. Check Document Relevance
# =========================================================

class RelevanceDecision(BaseModel):

    is_relevant: bool = Field(
        ...,
        description="True if the document is relevant to the question."
    )


relevance_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are judging document relevance for a RAG system.\n\n"

            "A document is relevant if it discusses the same "
            "topic or entity as the question.\n"

            "It does not need to contain the exact answer.\n\n"

            "Examples:\n"

            "- HR policies are relevant to questions about "
            "probation, termination and benefits.\n"

            "- Pricing documents are relevant to questions "
            "about refunds, trials and billing.\n"

            "- Company profile is relevant to questions about "
            "company culture, leadership and strategy.\n\n"

            "If unsure, return true."
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Document:\n{document}"
        )
    ]
)


relevance_chain = (
    relevance_prompt
    | llm.with_structured_output(RelevanceDecision)
)


def check_relevance(state: State):

    relevant_docs = []

    for doc in state.get("docs", []):

        decision = relevance_chain.invoke(
            {
                "question": state["question"],
                "document": doc.page_content
            }
        )

        if decision.is_relevant:
            relevant_docs.append(doc)

    return {
        "relevant_docs": relevant_docs
    }


def route_after_relevance(state: State):

    if state.get("relevant_docs"):
        return "generate_from_context"

    return "no_answer_found"


# =========================================================
# 8. Generate Answer From Documents
# =========================================================

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are a helpful business RAG chatbot.\n\n"

            "Answer the question using only the provided "
            "company document context.\n"

            "Do not use outside knowledge.\n"

            "Do not mention the context in your answer."
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Context:\n{context}"
        )
    ]
)


def generate_from_context(state: State):

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in state.get("relevant_docs", [])
    )

    if not context:

        return {
            "answer": "No answer found.",
            "context": ""
        }

    result = llm.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context
        )
    )

    return {
        "answer": result.content,
        "context": context
    }


# =========================================================
# 9. No Answer
# =========================================================

def no_answer_found(state: State):

    return {
        "answer": "No answer found.",
        "context": ""
    }


# =========================================================
# 10. IsSUP
# =========================================================

class ISSUPDecision(BaseModel):

    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]="no_support"

    evidence: List[str] = Field(
        default_factory=list
    )


issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are a strict factuality evaluator for a "
            "Self-RAG system.\n\n"

            "Check whether the ANSWER is supported by the "
            "provided CONTEXT.\n\n"

            "Rules:\n"

            "1. Use only the CONTEXT.\n"

            "2. Do not use outside knowledge.\n"

            "3. If every factual claim is supported, return "
            "'fully_supported'.\n"

            "4. If some claims are supported and some are not, "
            "return 'partially_supported'.\n"

            "5. If there is no meaningful support, return "
            "'no_support'.\n"

            "6. Paraphrasing is allowed.\n"

            "7. Provide short evidence."
        ),

        (
            "human",

            "QUESTION:\n{question}\n\n"
            "CONTEXT:\n{context}\n\n"
            "ANSWER:\n{answer}"
        )
    ]
)


issup_chain = (
    issup_prompt
    | llm.with_structured_output(ISSUPDecision)
)


def is_sup(state: State):

    decision = issup_chain.invoke(
        {
            "question": state["question"],
            "context": state.get("context", ""),
            "answer": state.get("answer", "")
        }
    )

    return {
        "issup": decision.issup,
        "evidence": decision.evidence
    }


# =========================================================
# 11. Route After IsSUP
# =========================================================

MAX_RETRIES = 3


def route_after_issup(state: State):

    if state.get("issup") == "fully_supported":
        return "is_use"

    if state.get("retries", 0) >= MAX_RETRIES:
        return "is_use"

    return "revise_answer"


# =========================================================
# 12. Revise Answer
# =========================================================

revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are a strict answer reviser.\n\n"

            "Rewrite the answer using ONLY the provided CONTEXT.\n"

            "Remove unsupported claims.\n"

            "Do not add outside information.\n"

            "Return only the revised answer."
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "Context:\n{context}"
        )
    ]
)


def revise_answer(state: State):

    result = llm.invoke(
        revise_prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )

    return {
        "answer": result.content,
        "retries": state.get("retries", 0) + 1
    }


# =========================================================
# 13. IsUSE
# =========================================================

class IsUSEDecision(BaseModel):

    isuse: Literal[
        "useful",
        "not_useful"
    ]

    reason: str = Field(
        description="Short reason in one line."
    )


isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "You are judging the usefulness of an answer.\n\n"

            "Decide whether the answer actually addresses "
            "the user's question.\n\n"

            "Useful means the answer directly answers the question.\n"

            "Not useful means the answer is generic, off-topic, "
            "or does not answer the question.\n\n"

            "Do not check factual grounding. IsSUP already does that."
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Answer:\n{answer}"
        )
    ]
)


isuse_chain = (
    isuse_prompt
    | llm.with_structured_output(IsUSEDecision)
)


def is_use(state: State):

    decision = isuse_chain.invoke(
        {
            "question": state["question"],
            "answer": state.get("answer", "")
        }
    )

    return {
        "isuse": decision.isuse,
        "use_reason": decision.reason
    }


# =========================================================
# 14. Rewrite Retrieval Query
# =========================================================

class RetrievalQuery(BaseModel):

    retrieval_query: str = Field(
        description="Improved query for internal company documents."
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",

            "Rewrite the question into a better query for "
            "retrieving information from internal company PDFs.\n\n"

            "Rules:\n"

            "- Keep it short, 6-16 words.\n"

            "- Keep important names and entities.\n"

            "- Add useful keywords.\n"

            "- Remove unnecessary words.\n"

            "- Do not answer the question."
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Previous Retrieval Query:\n{retrieval_query}\n\n"
            "Current Answer:\n{answer}"
        )
    ]
)


rewrite_chain = (
    rewrite_prompt
    | llm.with_structured_output(RetrievalQuery)
)


MAX_REWRITE_TRIES = 3


def rewrite_question(state: State):

    decision = rewrite_chain.invoke(
        {
            "question": state["question"],
            "retrieval_query": state.get(
                "retrieval_query",
                ""
            ),
            "answer": state.get(
                "answer",
                ""
            )
        }
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get(
            "rewrite_tries",
            0
        ) + 1,

        "docs": [],
        "relevant_docs": [],
        "context": ""
    }


# =========================================================
# 15. Route After IsUSE
# =========================================================

def route_after_isuse(state: State):

    if state.get("isuse") == "useful":
        return "END"

    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    return "rewrite_question"


# =========================================================
# 16. Build Graph
# =========================================================

graph = StateGraph(State)


graph.add_node(
    "decide_retrieval",
    decide_retrieval
)

graph.add_node(
    "generate_direct",
    generate_direct
)

graph.add_node(
    "retrieve",
    retrieve
)

graph.add_node(
    "check_relevance",
    check_relevance
)

graph.add_node(
    "generate_from_context",
    generate_from_context
)

graph.add_node(
    "no_answer_found",
    no_answer_found
)

graph.add_node(
    "is_sup",
    is_sup
)

graph.add_node(
    "revise_answer",
    revise_answer
)

graph.add_node(
    "is_use",
    is_use
)

graph.add_node(
    "rewrite_question",
    rewrite_question
)


# =========================================================
# 17. Graph Edges
# =========================================================

graph.add_edge(
    START,
    "decide_retrieval"
)


graph.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve"
    }
)


graph.add_edge(
    "generate_direct",
    END
)


graph.add_edge(
    "retrieve",
    "check_relevance"
)


graph.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found"
    }
)


graph.add_edge(
    "no_answer_found",
    END
)


graph.add_edge(
    "generate_from_context",
    "is_sup"
)


graph.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "is_use": "is_use",
        "revise_answer": "revise_answer"
    }
)


graph.add_edge(
    "revise_answer",
    "is_sup"
)


graph.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found"
    }
)


graph.add_edge(
    "rewrite_question",
    "retrieve"
)


# =========================================================
# 18. Compile
# =========================================================
memory=InMemorySaver()
app = graph.compile(checkpointer=memory)


# =========================================================
# 19. Function For Streamlit
# =========================================================

def run_self_rag(question: str,thread_id):

    result = app.invoke(
        {
            "question": question,
            "retrieval_query": "",
            "rewrite_tries": 0,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "issup": "no_support",
            "evidence": [],
            "retries": 0,
            "isuse": "not_useful",
            "use_reason": ""
        },
        config={
            "configurable":{
                "thread_id":"some-session-id"
            },
            "recursion_limit": 50
        }
    )

    return result