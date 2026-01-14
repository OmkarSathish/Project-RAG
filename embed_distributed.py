"""
Distributed Embedding Script for Sharded Qdrant Collections.

This script indexes documents into a sharded Qdrant cluster for
improved scalability and performance.
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_sharding import setup_distributed_collection

FILE = Path(__file__).parent / "nodejs.pdf"

load_dotenv()

# Single node configuration (for development)
# For production cluster, add multiple node URLs
QDRANT_NODES = [
    "http://localhost:6333",
]


async def embed_to_cluster():
    """Embed documents into sharded Qdrant cluster"""
    
    print("=" * 60)
    print("DISTRIBUTED DOCUMENT EMBEDDING")
    print("=" * 60)
    
    # 1. Load and split documents
    print("\n[1/4] Loading PDF document...")
    loader = PyPDFLoader(file_path=str(FILE))
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages")
    
    print("\n[2/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=350)
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    
    # 2. Setup sharded collection
    print("\n[3/4] Setting up distributed collection...")
    shard_manager = await setup_distributed_collection(
        nodes=QDRANT_NODES,
        collection_name="nodejs",
        preset="small",  # 1 shard for single node
        vector_size=3072,  # text-embedding-3-large dimension
    )
    
    # Get collection stats
    stats = await shard_manager.get_collection_stats("nodejs")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Shards: {len(stats['shards'])}")
    
    # 3. Generate embeddings and insert
    print("\n[4/4] Generating embeddings and inserting...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Process in batches
    batch_size = 50
    total_inserted = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in batch]
        embeddings = await embedding_model.aembed_documents(texts)
        
        # Prepare points
        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = i + j
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "page_content": chunk.page_content,
                    "page_label": chunk.metadata.get("page_label", str(chunk.metadata.get("page", 0))),
                    "source": chunk.metadata.get("source", str(FILE)),
                    "chunk_id": point_id,
                }
            })
        
        # Insert with shard routing
        await shard_manager.insert_points("nodejs", points)
        total_inserted += len(points)
        print(f"  Inserted {total_inserted}/{len(chunks)} chunks...")
    
    # Final stats
    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    final_stats = await shard_manager.get_collection_stats("nodejs")
    print(f"Total points: {final_stats['total_points']}")
    print("Shard distribution:")
    for shard in final_stats['shards']:
        print(f"  - Shard {shard.get('shard_id', 'N/A')}: {shard.get('points_count', 'N/A')} points ({shard.get('state', 'unknown')})")
    
    await shard_manager.close()


if __name__ == "__main__":
    asyncio.run(embed_to_cluster())
