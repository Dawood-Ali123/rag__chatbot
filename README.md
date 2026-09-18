# RAG Strategy Platform

An AI-powered RAG application that allows users to interact with their PDF documents using two different Retrieval-Augmented Generation strategies:

- CRAG (Corrective RAG)
- Self-RAG (Self-Reflective RAG)

The project demonstrates how different RAG strategies can improve document retrieval, answer generation, self-correction, and reliability.

---

## 🚀 Features

### 📄 PDF Document Processing
- Upload PDF documents
- Extract text from PDFs
- Split documents into smaller chunks
- Generate embeddings using HuggingFace embeddings
- Store and retrieve document chunks using FAISS

### 🔵 CRAG
Corrective RAG evaluates retrieved documents before generating an answer.

Workflow:

User Question  
→ Retrieve Documents  
→ Evaluate Retrieved Documents  
→ Correct Retrieval if Needed  
→ Web Search  
→ Refine Context  
→ Generate Answer

### 🟣 Self-RAG
Self-RAG evaluates its own retrieval and generated answers.

Workflow:

User Question  
→ Decide Whether Retrieval is Needed  
→ Retrieve Documents  
→ Check Relevance  
→ Generate Answer  
→ Check Answer Support (IsSUP)  
→ Check Usefulness (IsUSE)  
→ Refine / Retry if Needed  
→ Final Answer

### 🧠 LangGraph
Both RAG strategies are implemented as stateful LangGraph workflows.

### 💾 Short-Term Memory
LangGraph `InMemorySaver` is used for session-based graph state and thread management.

### 🔍 Web Search
CRAG can use Tavily Search when the retrieved documents are considered insufficient.

### 📊 RAG Evaluation
The project includes Ragas-based evaluation for:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

### 🔬 LangSmith
LangSmith is integrated for:

- LLM tracing
- LangGraph workflow observability
- Debugging
- Run monitoring
- Evaluation workflows

---

## 🏗️ Project Architecture

```text
RAG Strategy Platform
│
├── User
│   │
│   └── Upload PDF
│       │
│       ▼
│   Document Ingestion
│       │
│       ├── PyPDFLoader
│       ├── Text Splitter
│       ├── HuggingFace Embeddings
│       └── FAISS Vector Store
│
│
├── CRAG
│   │
│   ├── Retrieve
│   ├── Evaluate Documents
│   ├── Retrieval Decision
│   ├── Web Search (Tavily)
│   ├── Context Refinement
│   └── Answer Generation
│
│
└── Self-RAG
    │
    ├── Retrieval Decision
    ├── Retrieve
    ├── Relevance Check
    ├── Answer Generation
    ├── IsSUP
    ├── IsUSE
    ├── Answer Refinement
    └── Query Rewriting