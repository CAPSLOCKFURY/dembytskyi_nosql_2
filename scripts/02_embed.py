import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 64
TARGET_RECORDS = 5000  # Optimize to 5000 records to double execution speed on CPU

def main():
    print(f"Loading dataset from {INPUT_PARQUET}...")
    if not os.path.exists(INPUT_PARQUET):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PARQUET}. Please run 01_prepare_data.py first.")
    
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Original dataset size: {len(df)} papers.")
    
    # Slice to 5000 papers to speed up embedding generation on CPU
    if len(df) > TARGET_RECORDS:
        print(f"Slicing dataset to {TARGET_RECORDS} papers to optimize execution time...")
        df = df.head(TARGET_RECORDS).copy()
        df.to_parquet(INPUT_PARQUET, index=False)
        print(f"Updated {INPUT_PARQUET} with {len(df)} papers.")
        
    # 2. Prepare texts: title + " [SEP] " + abstract
    print("Preparing texts for encoding...")
    texts = []
    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        abstract = str(row["abstract"]).strip()
        texts.append(f"{title} [SEP] {abstract}")
    
    # 3. Load model
    print(f"Loading SentenceTransformer model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Generate embeddings
    print(f"Generating embeddings with batch_size={BATCH_SIZE} (this may take 8-10 minutes)...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    # 5. Print metrics
    total_texts = len(embeddings)
    embedding_dim = embeddings.shape[1]
    first_embedding_norm = np.linalg.norm(embeddings[0])
    
    print("\n--- Embedding Metrics ---")
    print(f"Total processed texts: {total_texts}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Norm of the first embedding: {first_embedding_norm:.6f}")
    print("-------------------------\n")
    
    # 6 & 7. Save to embeddings/embeddings.npy
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"Embeddings successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
