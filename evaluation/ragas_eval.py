import json
from pathlib import Path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rag.crag import run_crag
from rag.self_rag import run_self_rag
from ragas.llms import llm_factory
from ragas import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import Faithfulness,AnswerRelevancy,ContextRecall,ContextPrecision
from openai import OpenAI
from ragas import EvaluationDataset
import uuid
import os


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
evaluator_llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)





dataset_path = Path(
    r"C:\Users\User\OneDrive\Desktop\BLOG_Writig-agent\evaluation\test_dataset.json"
)

with open(dataset_path, "r", encoding="utf-8") as file:
    test_data = json.load(file)


results_file = Path("evaluation/results.json")

if results_file.exists():

    with open(results_file, "r", encoding="utf-8") as file:
        results = json.load(file)
else:
    results = []
# IDs that are already completed
completed_ids = {
    result["id"]
    for result in results
}
for test_case in test_data:

    # Skip already evaluated questions
    if test_case["id"] in completed_ids:
        print(
            f"Skipping {test_case['id']} - already evaluated"
        )

        continue
    question = test_case["question"]
    ground_truth = test_case["ground_truth"]
    print("\n" + "=" * 60)
    print("Question:", question)
    print("=" * 60)

    crag_result = run_crag(question,str(uuid.uuid4()))

    crag_answer = crag_result["answer"]

    crag_contexts = [
        doc.page_content
        for doc in crag_result["docs"]
    ]


    crag_dataset = EvaluationDataset.from_list([
        {
            "user_input": question,
            "response": crag_answer,
            "retrieved_contexts": crag_contexts,
            "reference": ground_truth
        }
    ])


    crag_evaluation = evaluate(
        crag_dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm)
        ],
        embeddings=evaluator_embeddings,
        show_progress=False
    )


    crag_score = crag_evaluation["faithfulness"][0]
    crag_relevancy=crag_evaluation['answer_relevancy'][0]
    crag_precision=crag_evaluation['context_precision'][0]
    crag_recall=crag_evaluation['context_recall'][0]



    self_result = run_self_rag(question,str(uuid.uuid4()))

    self_answer = self_result["answer"]

    self_contexts = [
        doc.page_content
        for doc in self_result["docs"]
    ]


    self_dataset = EvaluationDataset.from_list([
        {
            "user_input": question,
            "response": self_answer,
            "retrieved_contexts": self_contexts,
            "reference": ground_truth
        }
    ])


    self_evaluation = evaluate(
        self_dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm)

        ],
        embeddings=evaluator_embeddings,
        show_progress=False
    )


    self_score = self_evaluation["faithfulness"][0]
    self_relevancy = self_evaluation["answer_relevancy"][0]
    self_precision = self_evaluation["context_precision"][0]
    self_recall = self_evaluation["context_recall"][0]



    results.append({
        "id": test_case["id"],
        "question": question,
        "crag_faithfulness": crag_score,
        "crag_relevancy": crag_relevancy,
        "crag_precision": crag_precision,
        "crag_recall": crag_recall,

        "self_rag_faithfulness": self_score,
        "self_rag_relevancy": self_relevancy,
        "self_rag_precision": self_precision,
        "self_rag_recall": self_recall

    })


    with open(results_file, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            indent=4
        )
    print("CRAG:")
    print("  Faithfulness:", crag_score)
    print("  Relevancy:", crag_relevancy)
    print("  Context Precision:", crag_precision)
    print("  Context Recall:", crag_recall)

    print("Self-RAG:")
    print("  Faithfulness:", self_score)
    print("  Relevancy:", self_relevancy)
    print("  Context Precision:", self_precision)
    print("  Context Recall:", self_recall)

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)
print(f"Results saved to: {results_file}")