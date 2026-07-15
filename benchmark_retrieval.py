import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.text_processing.chunk import Chunk
from src.vector_store import IndexManager
from src.retrieval import RetrievalManager
from src.utils.logger import logger

class MockEmbeddingEngine:
    """Mock EmbeddingEngine to simulate CPU inference latency (30ms) without loading weights."""
    def __init__(self):
        self.dimension = 384
        
    def generate(self, texts):
        time.sleep(0.030)  # Simulate MiniLM CPU encoding time
        v = np.random.rand(384).astype(np.float32)
        v /= np.linalg.norm(v)
        return [v]

def ensure_indexed_docs(idx_manager: IndexManager):
    """Ensures at least some documents are indexed for benchmark runs."""
    if idx_manager.engine.total > 0:
        return
        
    logger.info("Vector index is empty. Populating with benchmarks dataset...")
    chunks = []
    vectors = []
    
    for i in range(100):
        c = Chunk(
            chunk_id=f"bench_chunk_{i:04d}",
            document_id="bench_doc",
            chunk_index=i,
            text=f"This is segment number {i} of the benchmark document text context.",
            metadata={"source_file": "benchmark.docx", "source_reference": f"Section {i}"},
            token_estimate=15,
            character_count=65
        )
        chunks.append(c)
        
        # Constant vector with random variation
        v = np.random.rand(384).astype(np.float32)
        v /= np.linalg.norm(v)
        vectors.append(v)
        
    idx_manager.add_vectors_and_metadata(np.array(vectors), chunks)

def run_benchmark():
    logger.info("=== Starting Retrieval Performance Benchmarks ===")
    
    idx_manager = IndexManager()
    ensure_indexed_docs(idx_manager)
    
    ret_manager = RetrievalManager(index_manager=idx_manager)
    
    # Inject mock embedding engine
    ret_manager.engine.embedding_manager.engine = MockEmbeddingEngine()
    
    # Warmer run
    ret_manager.search("Warmup test query")
    
    iterations = 20
    logger.info(f"Running search benchmarks over {iterations} iterations...")
    
    emb_times = []
    faiss_times = []
    db_times = []
    assembly_times = []
    total_times = []
    
    for i in range(iterations):
        query_text = f"Benchmark search query number {i}"
        
        start = time.time()
        res = ret_manager.search(query_text)
        total_dur = (time.time() - start) * 1000.0
        
        if res.success or res.reason == "NO_RELEVANT_CONTEXT":
            emb_times.append(res.latency_metrics.get("query_embedding_time_ms", 0.0))
            faiss_times.append(res.latency_metrics.get("faiss_search_time_ms", 0.0))
            db_times.append(res.latency_metrics.get("metadata_lookup_time_ms", 0.0))
            assembly_times.append(res.latency_metrics.get("context_assembly_time_ms", 0.0))
            total_times.append(total_dur)
            
    print("\n==================================================")
    print("           RETRIEVAL LATENCY PERFORMANCE          ")
    print("==================================================")
    print(f"Iterations: {iterations}")
    print(f"Total Vectors Indexed: {idx_manager.engine.total}")
    print("--------------------------------------------------")
    print(f"1. Query Embedding Generation Time:")
    print(f"   Average: {np.mean(emb_times):.2f} ms")
    print(f"   Min/Max: {np.min(emb_times):.2f} ms / {np.max(emb_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"2. FAISS Raw Index Search Time:")
    print(f"   Average: {np.mean(faiss_times):.2f} ms")
    print(f"   Min/Max: {np.min(faiss_times):.2f} ms / {np.max(faiss_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"3. SQLite Metadata Lookup Time:")
    print(f"   Average: {np.mean(db_times):.2f} ms")
    print(f"   Min/Max: {np.min(db_times):.2f} ms / {np.max(db_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"4. Context Package Assembly Time:")
    print(f"   Average: {np.mean(assembly_times):.2f} ms")
    print(f"   Min/Max: {np.min(assembly_times):.2f} ms / {np.max(assembly_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"5. Total Pipeline Retrieval Time:")
    print(f"   Average: {np.mean(total_times):.2f} ms")
    print(f"   Min/Max: {np.min(total_times):.2f} ms / {np.max(total_times):.2f} ms")
    print("==================================================\n")

if __name__ == "__main__":
    run_benchmark()
