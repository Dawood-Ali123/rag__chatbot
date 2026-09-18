from typing import List, TypedDict, Literal
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
VECTORSTORE_DIR = "data/vectorstore"
def get_reteriver():
    embeddings = HuggingFaceEmbeddings(
         model_name="BAAI/bge-small-en-v1.5"
        )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
        )
model = ChatGroq(model="openai/gpt-oss-20b")
UPPER_TH = 0.7
LOWER_TH = 0.3

class State(TypedDict, total=False):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    web_docs: List[Document]
    web_query: str
    refined_context: str
    answer: str

def retrieve(state: State):
    q = state["question"]
    retriever=get_reteriver()
    docs = retriever.invoke(q)
    return {
        "docs": docs
    }
class DocEvalScore(BaseModel):
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given one retrieved chunk and a question.\n"
            "Return a relevance score between 0.0 and 1.0.\n\n"
            "1.0: The chunk is sufficient to answer the question fully "
            "or mostly.\n"
            "0.0: The chunk is irrelevant.\n\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Return structured output only."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Chunk:\n{chunk}"
        )
    ]
)


doc_eval_chain = (
    doc_eval_prompt
    | model.with_structured_output(DocEvalScore)
)


def eval_each_doc(state: State):
    question = state["question"]
    scores: List[float] = []
    good_docs: List[Document] = []
    for doc in state.get("docs", []):
        result = doc_eval_chain.invoke(
            {
                "question": question,
                "chunk": doc.page_content
            }
        )

        scores.append(result.score)
        if result.score > LOWER_TH:
            good_docs.append(doc)
    # Correct Retrieval
    if any(score > UPPER_TH for score in scores):

        return {
            "good_docs": good_docs,
            "verdict": "Correct",
            "reason": (
                f"At least one retrieved chunk scored "
                f"> {UPPER_TH}."
            )
        }

    # Incorrect Retrieval

    if scores and all(score < LOWER_TH for score in scores):

        return {
            "good_docs": [],
            "verdict": "Incorrect",
            "reason": (
                f"All retrieved chunks scored "
                f"< {LOWER_TH}."
            )
        }

    # Ambiguous Retrieval
    return {
        "good_docs": good_docs,
        "verdict": "Ambiguous",
        "reason": (
            f"Mixed relevance signals. No chunk scored "
            f"> {UPPER_TH}, but not all were "
            f"< {LOWER_TH}."
        )
    }

def decompose_to_sentence(text: str) -> List[str]:

    text = re.sub(r"\s+"," ",text).strip()
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )
    return [
        sentence.strip() for sentence in sentences if len(sentence.strip()) > 20
    ]

class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly "
            "helps answer the question.\n"
            "Do not use outside knowledge.\n"
            "Return structured output only."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Sentence:\n{sentence}"
        )
    ]
)


filter_chain = (
    filter_prompt
    | model.with_structured_output(KeepOrDrop)
)

def refine(state: State):
    question = state["question"]
    verdict = state.get("verdict")
    if verdict == "Correct":
        docs_to_use = state.get(
            "good_docs",
            []
        )
    elif verdict == "Incorrect":
        docs_to_use = state.get(
            "web_docs",
            []
        )
    else:
        docs_to_use = (
            state.get("docs", [])
            + state.get("web_docs", [])
        )
    context = "\n\n".join(
        doc.page_content
        for doc in docs_to_use
    ).strip()


    sentences = decompose_to_sentence(
        context
    )


    kept_sentences: List[str] = []


    for sentence in sentences:

        result = filter_chain.invoke(
            {
                "question": question,
                "sentence": sentence
            }
        )

        if result.keep:

            kept_sentences.append(
                sentence
            )


    refined_output = "\n".join(
        kept_sentences
    ).strip()


    return {
        "strips": sentences,
        "kept_strips": kept_sentences,
        "refined_context": refined_output
    }


# =========================================================
# 10. Web Query Rewriting
# =========================================================

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question into a web search query "
            "composed of useful keywords.\n\n"
            "Rules:\n"
            "- Keep it short, around 6-14 words.\n"
            "- If the question implies recency such as recent, "
            "latest, last week, or last month, include an "
            "appropriate recency constraint.\n"
            "- Do not answer the question.\n"
            "Return structured output only."
        ),
        (
            "human",
            "Question:\n{question}"
        )
    ]
)


rewrite_chain = (
    rewrite_prompt
    | model.with_structured_output(WebQuery)
)

def rewrite_query_node(state: State):
    result = rewrite_chain.invoke(
        {
            "question": state["question"]
        }
    )

    return {
        "web_query": result.query
    }


# 11. Tavily Web Search

tavily = TavilySearchResults(
    max_results=5
)

def web_search(state: State):
    query = (
        state.get("web_query")
        or state["question"]
    )
    results = tavily.invoke(
        {
            "query": query
        }
    )
    web_docs: List[Document] = []
    for result in results or []:
        title = result.get(
            "title",
            ""
        )
        url = result.get(
            "url",
            ""
        )
        content = (
            result.get("content")
            or result.get("snippet")
            or ""
        )
        text = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )
        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title
                }
            )
        )


    return {
        "web_docs": web_docs
    }



# 12. Answer Generation

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor.\n"
            "Answer only using the provided refined bullets.\n"
            "If the bullets are empty or insufficient, say:\n"
            "'I don't know based on the provided sources.'"
        ),

        (
            "human",

            "Question:\n{question}\n\n"
            "Refined Context:\n{refined_context}"
        )
    ]
)


def generate(state: State):
    chain = answer_prompt | model
    result = chain.invoke(
        {
            "question": state["question"],
            "refined_context": state.get(
                "refined_context",
                ""
            )
        }
    )


    return {
        "answer": result.content
    }



# 13. Route After Evaluation


def route_after_eval(state: State):

    if state.get("verdict") == "Correct":

        return "refine"

    return "rewrite_query"


# =========================================================
# 14. Build Graph
# =========================================================

graph = StateGraph(State)


graph.add_node(
    "retrieve",
    retrieve
)

graph.add_node(
    "eval_each_doc",
    eval_each_doc
)

graph.add_node(
    "refine",
    refine
)

graph.add_node(
    "generate",
    generate
)

graph.add_node(
    "rewrite_query",
    rewrite_query_node
)

graph.add_node(
    "web_search",
    web_search
)


# =========================================================
# 15. Graph Edges
# =========================================================

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "eval_each_doc"
)


graph.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query"
    }
)


graph.add_edge(
    "rewrite_query",
    "web_search"
)

graph.add_edge(
    "web_search",
    "refine"
)

graph.add_edge(
    "refine",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

# 16. Compile
memory=InMemorySaver()
app = graph.compile(checkpointer=memory)
# 17. Reusable CRAG Function

def run_crag(question: str,thread_id):

    result = app.invoke(
        {
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "web_docs": [],
            "web_query": "",
            "refined_context": "",
            "answer": ""
        },
        config={"configurable":{
                "thread_id":"some_session_id"
            },
            "recursion_limit":50
        }
    )

    return result