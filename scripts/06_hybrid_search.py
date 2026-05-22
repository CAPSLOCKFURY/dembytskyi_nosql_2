import os
import re
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5  # For final display
RETRIEVAL_K = 50  # We retrieve a larger number of candidates for RRF fusion
RRF_K = 60  # Hyperparameter k for RRF formula: 1 / (RRF_K + rank)
INPUT_PARQUET = "data/arxiv_subset.parquet"

def tokenize(text):
    # Split text into lowercase words/tokens, removing punctuation
    return [token for token in re.findall(r'\w+', text.lower()) if token]

class HybridSearcher:
    def __init__(self, df, pinecone_index, model):
        self.df = df
        self.index = pinecone_index
        self.model = model
        
        print("Building local BM25 index...")
        # Prepare BM25 corpus (combine title and abstract)
        self.corpus = []
        for _, row in self.df.iterrows():
            text = f"{row['title']} {row['abstract']}"
            self.corpus.append(tokenize(text))
            
        self.bm25 = BM25Okapi(self.corpus)
        print("BM25 index built successfully.")

    def bm25_search(self, query, top_k=RETRIEVAL_K):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of highest scoring items
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            row = self.df.iloc[idx]
            results.append({
                "id": f"paper_{idx}",
                "title": row["title"],
                "abstract": row["abstract"],
                "authors": row["authors"],
                "year": int(row["year"]),
                "category": row["category"],
                "score": float(scores[idx]),
                "rank": rank + 1
            })
        return results

    def vector_search(self, query, top_k=RETRIEVAL_K):
        # Generate query embedding
        query_emb = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Query Pinecone
        res = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for rank, match in enumerate(res.get("matches", [])):
            meta = match.get("metadata", {})
            results.append({
                "id": match["id"],
                "title": meta.get("title"),
                "abstract": meta.get("abstract"),
                "authors": meta.get("authors"),
                "year": int(meta.get("year", 0)),
                "category": meta.get("category"),
                "score": float(match["score"]),
                "rank": rank + 1
            })
        return results

    def hybrid_search(self, query, top_k=TOP_K, retrieval_k=RETRIEVAL_K, rrf_k=RRF_K):
        # 1. Fetch search results from both search methods
        bm25_res = self.bm25_search(query, top_k=retrieval_k)
        vector_res = self.vector_search(query, top_k=retrieval_k)
        
        # 2. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_details = {}
        
        # Helper to update RRF score
        def update_rrf(doc_list):
            for doc in doc_list:
                doc_id = doc["id"]
                rank = doc["rank"]
                
                # Formula: 1 / (rrf_k + rank)
                contribution = 1.0 / (rrf_k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution
                
                if doc_id not in doc_details:
                    doc_details[doc_id] = {
                        "id": doc_id,
                        "title": doc["title"],
                        "abstract": doc["abstract"],
                        "authors": doc["authors"],
                        "year": doc["year"],
                        "category": doc["category"]
                    }
                    
        update_rrf(bm25_res)
        update_rrf(vector_res)
        
        # 3. Sort doc_ids by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 4. Construct final results
        final_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            details = doc_details[doc_id]
            final_results.append({
                **details,
                "score": score,
                "rank": rank + 1
            })
            
        return final_results

def print_search_results(results, title_text):
    print(f"\n--- {title_text} (Top {len(results)}) ---")
    if not results:
        print("No results found.")
        return
    for i, doc in enumerate(results):
        print(f" {i+1}. [Score: {doc['score']:.5f}] {doc['title']}")
        print(f"    ID: {doc['id']} | Category: {doc['category']} | Year: {doc['year']}")
        print(f"    Authors: {doc['authors']}")
        print(f"    Abstract snippet: {doc['abstract'][:150]}...")
        print()

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
        
    # Load dataset
    print("Loading local parquet dataset...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise FileNotFoundError(f"Pinecone index '{INDEX_NAME}' does not exist. Please run 03_load_to_pinecone.py first.")
    
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    # Instantiate Hybrid Searcher
    searcher = HybridSearcher(df, index, model)
    
    # 5. Run the 3 demonstration queries
    demo_queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text"
    ]
    
    for q in demo_queries:
        print("\n" + "="*80)
        print(f"TEST QUERY: '{q}'")
        print("="*80)
        
        bm25_res = searcher.bm25_search(q, top_k=TOP_K)
        vector_res = searcher.vector_search(q, top_k=TOP_K)
        hybrid_res = searcher.hybrid_search(q, top_k=TOP_K, retrieval_k=RETRIEVAL_K, rrf_k=RRF_K)
        
        print_search_results(bm25_res, "BM25 Search Results")
        print_search_results(vector_res, "Vector (Pinecone) Search Results")
        print_search_results(hybrid_res, "Hybrid (BM25 + Vector RRF) Search Results")

if __name__ == "__main__":
    main()
