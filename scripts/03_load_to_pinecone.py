import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200  # Pinecone recommends batches up to 200 vectors

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")

    # 1. Initialize Pinecone client
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=api_key)

    # Create index if it does not exist
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating serverless index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    # Connect to the index
    print(f"Connecting to index '{INDEX_NAME}'...")
    index = pc.Index(INDEX_NAME)

    # 2. Load dataset and embeddings
    print(f"Loading data from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loading embeddings from {INPUT_EMBEDDINGS}...")
    embeddings = np.load(INPUT_EMBEDDINGS)

    if len(df) != len(embeddings):
        raise ValueError(f"Mismatch: parquet has {len(df)} rows, but embeddings has {len(embeddings)} vectors.")

    # 3. Prepare data for loading
    print("Preparing and uploading records in batches...")
    records = []
    for idx, row in df.iterrows():
        # Ensure values are float64/float32 and converted to list of python floats
        vector = embeddings[idx].tolist()
        
        # Build metadata dictionary
        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"])[:500],
            "authors": str(row["authors"])[:200],
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        records.append({
            "id": f"paper_{idx}",
            "values": vector,
            "metadata": metadata
        })

    # 4. Upload in batches
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Uploading to Pinecone"):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

    # 5. Output total count of vectors
    print("Fetching index statistics...")
    stats = index.describe_index_stats()
    print("\n--- Pinecone Index Statistics ---")
    print(f"Index Name: {INDEX_NAME}")
    print(f"Total vector count: {stats['total_vector_count']}")
    print(f"Dimensions: {stats['dimension']}")
    print("---------------------------------\n")

if __name__ == "__main__":
    main()
