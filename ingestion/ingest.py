from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"

# embedding model
embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def ingest_document():
    documents=[]
    pdf_files=list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError("NO PDF documents found")

    #load pdf
    for pdf_file in pdf_files:
        loader=PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    splitter=RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks=splitter.split_documents(documents)

    for chunk in chunks:
        chunk.page_content=(
            chunk.page_content
            .encode("utf-8","ignore")
            .decode("utf-8",'ignore')
        )
    vectorestore=FAISS.from_documents(
        chunks,
        embeddings
    )
    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    vectorestore.save_local(str(VECTORSTORE_DIR))
    return {
        'documents':len(pdf_files),
        "chunks":len(chunks)
    }