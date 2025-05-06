from llama_stack_client import LlamaStackClient
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import argparse

import pathlib, os
import uuid
from typing import Dict, List, Set

import matplotlib.pyplot as plt

from llama_stack_client.types import Document
from llama_stack.apis.tools import RAGQueryConfig
from llama_stack_client import LlamaStackClient

def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client

client = LlamaStackClient(base_url="http://localhost:8321") # or create_library_from_client()

def parse_args():
    parser = argparse.ArgumentParser(description="Search script with database and search-modes")

    parser.add_argument(
        "--vector_db_id",
        type=str,
        required=False,
        help="Name of the database provider id to use",
        default="faiss"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        required=False,
        help="The beir datasets to run the benchmark against seperated by a space",
        default=["scifact"]
    )

    parser.add_argument(
        "--limits",
        type=int,
        nargs='+',
        required=False,
        help="The document limits seperated by a space",
        default=[100, 140, 180, 220]
    )

    return parser.parse_args()

def compute_recall_metrics(gold_ids: Set[str], retrieved_ids: List[str]) -> Dict[str, float]:
    def hit_at_k(k):
        return 1.0 if any(id in gold_ids for id in retrieved_ids[:k]) else 0.0

    return {
        "Recall@1": hit_at_k(1),
        "Recall@3": hit_at_k(3),
        "Recall@5": hit_at_k(5),
    }

def compute_precision_metrics(gold_ids: Set[str], retrieved_ids: List[str]) -> Dict[str, float]:
    def precision_at_k(k):
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        correct = sum(1 for id in top_k if id in gold_ids)
        return correct / k

    return {
        "Precision@1": precision_at_k(1),
        "Precision@3": precision_at_k(3),
        "Precision@5": precision_at_k(5),
    }

def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def prepare_data(corpus: dict, queries: dict, qrels: dict, limit:int):
    qa_pairs = []
    documents = []

    # Match query with related ids
    for qid, docs in qrels.items():
        query_text = queries.get(qid)
        if query_text:
            for doc_id in docs.keys():
                qa_pairs.append((query_text, doc_id))
    
    # Only insert documents that HAVE associated queries
    for idx, item in corpus.items():
        for _, rel_id in qa_pairs:
            if idx == rel_id:
                title = item.get('title', '')
                text = item.get('text', '')
                documents.append(Document(document_id=idx, content=text, metadata={"title": title}))

    return qa_pairs[:limit], documents[:limit]


def load_dataset(dataset_name: str):
    #### Download zip dataset and unzip the dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where the dataset has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    return corpus, queries, qrels

def evaluate_retriever(dataset_name: str, chunk_size: int, embedding_dimension: int, limit: int, provider_id: str):
    corpus, queries, qrels = load_dataset(dataset_name)
    qa_pairs, documents = prepare_data(corpus,queries,qrels,limit)

    vector_db_id = f"{dataset_name}-db-{uuid.uuid4().hex}"

    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=embedding_dimension,
        provider_id=provider_id,
    )

    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=chunk_size,
    )

    recall_buckets = {"Recall@1": [], "Recall@3": [], "Recall@5": []}
    precision_buckets = {"Precision@1": [], "Precision@3": [], "Precision@5": []}

    for idx, (question, ground_truth) in enumerate(qa_pairs):
        query_config = RAGQueryConfig(max_chunks=5, mode="vector").model_dump()
        result = client.tool_runtime.rag_tool.query(
            vector_db_ids=[vector_db_id],
            content=question,
            query_config=query_config,
        )


        retrieved_ids = result.metadata.get("document_ids", [])

        # Evaluation code mostly untouched except for using ids for finding matches instead of titles.
        recall_scores = compute_recall_metrics(ground_truth, retrieved_ids)
        precision_scores = compute_precision_metrics(ground_truth, retrieved_ids)

        for key in recall_buckets.keys():
            recall_buckets[key].append(recall_scores[key])
        for key in precision_buckets.keys():
            precision_buckets[key].append(precision_scores[key])

    avg_recall = {key: sum(scores) / len(scores) for key, scores in recall_buckets.items()}
    avg_precision = {key: sum(scores) / len(scores) for key, scores in precision_buckets.items()}
    return avg_recall, avg_precision

def main():
    args = parse_args()
    import time # DELETE

    # msmarco is too large for local runs
    beir_datasets= args.datasets # List of datasets to run benchmarks on
    chunk_size = 256
    embedding_dim = 64
    limits = [100, 140, 180, 220]
    provider_id = args.vector_db_id

    total_time=0
    for dataset in beir_datasets:
        recall_results = {"Recall@1": [], "Recall@3": [], "Recall@5": []}
        precision_results = {"Precision@1": [], "Precision@3": [], "Precision@5": []}
        f1_results = {"F1@1": [], "F1@3": [], "F1@5": []}

        start_time = time.time() # DELETE
        for limit in limits:
            print(f"Running bench on {dataset} for limit: {limit}")
            
            avg_recall, avg_precision = evaluate_retriever(dataset_name=dataset, chunk_size=chunk_size, embedding_dimension=embedding_dim, limit=limit, provider_id=provider_id)
            for key in recall_results:
                recall_results[key].append(avg_recall[key])
            for key in precision_results:
                precision_results[key].append(avg_precision[key])

            for k in ["1", "3", "5"]:
                prec = avg_precision[f"Precision@{k}"]
                rec = avg_recall[f"Recall@{k}"]
                f1 = compute_f1(prec, rec)
                f1_results[f"F1@{k}"].append(f1)

        plt.figure()
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Recall plot
        for key in recall_results:
            axes[0].plot(limits, recall_results[key], marker="o", label=key)
            for i, value in enumerate(recall_results[key]):
                axes[0].text(limits[i], value + 0.01, f"{limits[i]}", ha='center', fontsize=8)

        axes[0].set_xlabel("Dataset Size (limit)")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall@K vs Dataset Size\n(embedding_dim=64, chunk_size=256)")
        axes[0].legend()
        axes[0].grid(True)

        # Precision plot
        for key in precision_results:
            axes[1].plot(limits, precision_results[key], marker="o", label=key)
            for i, value in enumerate(precision_results[key]):
                axes[1].text(limits[i], value + 0.01, f"{limits[i]}", ha='center', fontsize=8)

        axes[1].set_xlabel("Dataset Size (limit)")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision@K vs Dataset Size\n(embedding_dim=64, chunk_size=256)")
        axes[1].legend()
        axes[1].grid(True)

        # F1 plot
        for key in f1_results:
            axes[2].plot(limits, f1_results[key], marker="o", label=key)
            for i, value in enumerate(f1_results[key]):
                axes[2].text(limits[i], value + 0.01, f"{limits[i]}", ha='center', fontsize=8)

        axes[2].set_xlabel("Dataset Size (limit)")
        axes[2].set_ylabel("F1 Score")
        axes[2].set_title("F1@K vs Dataset Size\n(embedding_dim=64, chunk_size=256)")
        axes[2].legend()
        axes[2].grid(True)

        plt.title(f"RAG Information Retrieval Benchmark: {dataset}")
        plt.tight_layout()

        print(f"Saving figure as benchmark-{provider_id}-{dataset}")
        plt.savefig(f"benchmark-{provider_id}-{dataset}")

        time_this_run = time.time() - start_time # DELETE
        total_time = total_time + time_this_run # DELETE
        print("--- %s seconds ---" % (time_this_run)) # DELETE
    print(f"TOTAL TIME: {total_time}") # DELETE

if __name__ == "__main__":
    main()