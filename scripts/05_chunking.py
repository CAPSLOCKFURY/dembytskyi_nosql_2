import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
INPUT_PARQUET = "data/arxiv_subset.parquet"
BATCH_SIZE = 100

def fixed_size_chunking(text, chunk_size=50, overlap=10):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def semantic_chunking(text, max_words=50):
    # Split text into sentences using regex (split on punctuation followed by whitespace)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())
        
        # If adding this sentence doesn't exceed max_words, OR if the current chunk is empty
        # (to avoid getting stuck if a single sentence exceeds max_words)
        if current_word_count + sentence_words <= max_words or not current_chunk:
            current_chunk.append(sentence)
            current_word_count += sentence_words
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def create_and_get_index(pc, index_name):
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating serverless index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)

def process_and_upload_chunks(pc, index_name, papers, chunking_type, model):
    index = create_and_get_index(pc, index_name)
    
    records = []
    print(f"Generating chunks and embeddings for strategy: {chunking_type}...")
    
    for _, row in papers.iterrows():
        abstract = str(row["abstract"]).strip()
        arxiv_id = str(row["id"])
        title = str(row["title"]).strip()
        year = int(row["year"])
        category = str(row["category"])
        
        if chunking_type == "fixed":
            chunks = fixed_size_chunking(abstract, chunk_size=40, overlap=10)
        else:
            chunks = semantic_chunking(abstract, max_words=45)
            
        for chunk_idx, chunk_text in enumerate(chunks):
            # Specter2 requires title [SEP] text for embedding
            text_to_encode = f"{title} [SEP] {chunk_text}"
            emb = model.encode(text_to_encode, normalize_embeddings=True).tolist()
            
            unique_id = f"chunk_{chunking_type}_{arxiv_id}_{chunk_idx}"
            metadata = {
                "arxiv_id": arxiv_id,
                "title": title,
                "chunk_text": chunk_text,
                "chunk_index": chunk_idx + 1,
                "year": year,
                "category": category,
                "chunk_type": chunking_type
            }
            records.append({
                "id": unique_id,
                "values": emb,
                "metadata": metadata
            })
            
    print(f"Uploading {len(records)} {chunking_type} chunks to '{index_name}'...")
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Uploading {chunking_type}"):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        
    stats = index.describe_index_stats()
    print(f"Completed! Total vectors in '{index_name}': {stats['total_vector_count']}")
    return index

def run_queries(index, model, queries, title):
    print(f"\n==========================================")
    print(f"--- Results for Index: {title} ---")
    print(f"==========================================")
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        q_emb = model.encode(q, normalize_embeddings=True).tolist()
        results = index.query(vector=q_emb, top_k=5, include_metadata=True)
        
        for i, match in enumerate(results.get("matches", [])):
            meta = match.get("metadata", {})
            score = match.get("score", 0.0)
            print(f"  {i+1}. [Score: {score:.4f}] Paper: '{meta.get('title')}' (Chunk {meta.get('chunk_index')})")
            print(f"     Text: {meta.get('chunk_text')[:180]}...")
            print()

def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
        
    pc = Pinecone(api_key=api_key)
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Load dataset
    print(f"Loading dataset from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    
    # Select 30 papers with the longest abstracts
    df["abstract_len"] = df["abstract"].str.len()
    top_30_longest = df.sort_values(by="abstract_len", ascending=False).head(30)
    print(f"Selected 30 papers with longest abstracts. Length range: {top_30_longest['abstract_len'].min()} to {top_30_longest['abstract_len'].max()} characters.")
    
    # 2. Process and upload Fixed-size chunking
    index_fixed = process_and_upload_chunks(pc, INDEX_FIXED, top_30_longest, "fixed", model)
    
    # 3. Process and upload Semantic chunking
    index_semantic = process_and_upload_chunks(pc, INDEX_SEMANTIC, top_30_longest, "semantic", model)
    
    # 6. Run Search and compare
    test_queries = [
        "methods for mapping and modeling dark matter distribution",
        "quantum error correction codes in computing systems"
    ]
    
    run_queries(index_fixed, model, test_queries, "Fixed-Size Chunking")
    run_queries(index_semantic, model, test_queries, "Semantic Chunking")

if __name__ == "__main__":
    main()
