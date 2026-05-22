import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

def encode_query(model, query_text):
    # Specter2 was trained on [title] [SEP] [abstract] or just queries.
    # Standard sentence-transformers encode returns a numpy array.
    # We normalize the query embedding to match the document embeddings.
    emb = model.encode(query_text, normalize_embeddings=True)
    return emb.tolist(), emb

def print_results(results, title_text):
    print(f"\n=== {title_text} ===")
    if not results or not results.get("matches"):
        print("No matches found.")
        return
    for i, match in enumerate(results["matches"]):
        meta = match.get("metadata", {})
        score = match.get("score", 0.0)
        print(f"{i+1}. [Score: {score:.4f}] {meta.get('title')}")
        print(f"   ID: {meta.get('arxiv_id')} | Category: {meta.get('category')} | Year: {meta.get('year')}")
        print(f"   Authors: {meta.get('authors')}")
        print(f"   Abstract: {meta.get('abstract')[:180]}...")
        print()

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")

    # Load resources
    print("Initializing Pinecone and loading model...")
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        raise FileNotFoundError(f"Pinecone index '{INDEX_NAME}' does not exist. Please run 03_load_to_pinecone.py first.")
        
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    # Load full dataset for local abstract retrieval if needed (or just use metadata)
    df = pd.read_parquet(INPUT_PARQUET)
    
    # 3. Clean Semantic Search
    query = "teaching machines to recognize objects in pictures"
    print(f"\nRunning vanilla semantic search for query: '{query}'")
    query_list, query_np = encode_query(model, query)
    
    results = index.query(
        vector=query_list,
        top_k=TOP_K,
        include_metadata=True
    )
    print_results(results, "Vanilla Semantic Search Results")

    # 4. Search with filtering
    print("\n--- Search with Metadata Filtering ---")
    
    # Example A: reinforcement learning in last 5 years (year >= 2021) and category cs.LG
    # Note: Since our subset only contains papers from 2007, this will likely return 0 matches.
    # We will demonstrate this, but also show a working example with year >= 2005 and category hep-ph.
    print("\nRunning Filter A (category = cs.LG, year >= 2021)...")
    results_filter_a = index.query(
        vector=query_list,
        top_k=TOP_K,
        filter={
            "category": {"$eq": "cs.LG"},
            "year": {"$gte": 2021}
        },
        include_metadata=True
    )
    print_results(results_filter_a, "Filter A Results (cs.LG, year >= 2021)")
    
    print("\nRunning Alternate Filter A (category = hep-ph, year >= 2005) to demonstrate working filter...")
    results_filter_a_alt = index.query(
        vector=query_list,
        top_k=TOP_K,
        filter={
            "category": {"$eq": "hep-ph"},
            "year": {"$gte": 2005}
        },
        include_metadata=True
    )
    print_results(results_filter_a_alt, "Alternate Filter A Results (hep-ph, year >= 2005)")

    # Example B: older articles (before 2015), any category
    print("\nRunning Filter B (year < 2015)...")
    results_filter_b = index.query(
        vector=query_list,
        top_k=TOP_K,
        filter={
            "year": {"$lt": 2015}
        },
        include_metadata=True
    )
    print_results(results_filter_b, "Filter B Results (year < 2015)")

    # 5. Local similarity comparison
    print("\n--- Local Metric Comparison (Cosine vs Dot Product vs L2) ---")
    if not os.path.exists(INPUT_EMBEDDINGS):
        raise FileNotFoundError(f"Local embeddings not found at {INPUT_EMBEDDINGS}. Please run 02_embed.py first.")
    
    embeddings = np.load(INPUT_EMBEDDINGS)
    
    # Calculate similarities
    # 1. Cosine similarity
    # Cosine = (A . B) / (||A|| * ||B||)
    # Since embeddings and query are already normalized, cosine = dot product
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_np)
    
    # Just to be mathematically explicit and generic:
    cosine_sim = np.dot(embeddings, query_np) / (norms * query_norm)
    
    # 2. Dot product
    dot_prod = np.dot(embeddings, query_np)
    
    # 3. L2 distance
    l2_dist = np.linalg.norm(embeddings - query_np, axis=1)
    
    # Get top-5 indices
    top_cosine = np.argsort(cosine_sim)[::-1][:TOP_K]
    top_dot = np.argsort(dot_prod)[::-1][:TOP_K]
    top_l2 = np.argsort(l2_dist)[:TOP_K]  # for L2, smallest is best
    
    # Display comparison
    print("\nTop 5 by Cosine Similarity:")
    for i, idx in enumerate(top_cosine):
        row = df.iloc[idx]
        print(f"  {i+1}. [Sim: {cosine_sim[idx]:.4f}] {row['title']} (Year: {row['year']}, Cat: {row['category']})")
        
    print("\nTop 5 by Dot Product:")
    for i, idx in enumerate(top_dot):
        row = df.iloc[idx]
        print(f"  {i+1}. [Val: {dot_prod[idx]:.4f}] {row['title']} (Year: {row['year']}, Cat: {row['category']})")
        
    print("\nTop 5 by L2 Distance (lower is better):")
    for i, idx in enumerate(top_l2):
        row = df.iloc[idx]
        print(f"  {i+1}. [Dist: {l2_dist[idx]:.4f}] {row['title']} (Year: {row['year']}, Cat: {row['category']})")
        
    print("\nLocal comparison complete.")

if __name__ == "__main__":
    main()
