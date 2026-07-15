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

from src.retrieval.metadata import RetrievalChunk, RetrievalResult
from src.llm import LLMManager
from src.utils.logger import logger

class MockOllamaClient:
    """Mock HTTP client simulating CPU model execution latency (500ms)."""
    def __init__(self, latency_sec: float = 0.5):
        self.latency_sec = latency_sec
        self.base_url = "http://localhost:11434"
        
    def is_server_running(self):
        return True
        
    def is_model_installed(self, model_name):
        return True
        
    def generate(self, model_name, prompt):
        time.sleep(self.latency_sec)  # Simulate CPU model generation delay
        response_text = "The artificial intelligence pipeline processes query vectors to retrieve matching contexts."
        stats = {
            "prompt_eval_count": len(prompt) // 4,  # Approx tokens
            "eval_count": len(response_text) // 4
        }
        return response_text, stats

def run_benchmark():
    logger.info("=== Starting LLM Generation Performance Benchmarks ===")
    
    # 1. Setup Retrieval chunks of context
    chunks = []
    for i in range(3):
        c = RetrievalChunk(
            chunk_id=f"c_{i}",
            document_id="bench_doc",
            source_file="manual.pdf",
            source_reference=f"Page {i+1}",
            similarity_score=0.90 - (i * 0.05),
            confidence="High" if i == 0 else "Medium",
            chunk_text=f"The artificial intelligence pipeline processes query vectors to retrieve matching contexts. Segment details for page {i+1}."
        )
        chunks.append(c)
        
    ret_res = RetrievalResult(
        success=True,
        reason="SUCCESS",
        message="Retrieval OK",
        query="Explain how the artificial intelligence pipeline retrieves context.",
        language="en",
        language_confidence=1.0,
        retrieved_chunks=chunks,
        combined_context=" ".join(c.chunk_text for c in chunks),
        total_chunks=3
    )
    
    manager = LLMManager()
    manager.client = MockOllamaClient(latency_sec=0.5)
    
    iterations = 5
    logger.info(f"Running LLM benchmarks over {iterations} iterations...")
    
    prompt_times = []
    inf_times = []
    fmt_times = []
    total_times = []
    prompt_sizes = []
    context_sizes = []
    response_lens = []
    
    for i in range(iterations):
        start = time.time()
        ans = manager.generate_grounded_answer(ret_res)
        total_dur = (time.time() - start) * 1000.0
        
        if ans.success:
            prompt_times.append(ans.latency_metrics.get("prompt_construction_time_ms", 0.0))
            inf_times.append(ans.latency_metrics.get("inference_time_ms", 0.0))
            fmt_times.append(ans.latency_metrics.get("formatting_time_ms", 0.0))
            total_times.append(total_dur)
            
            # Record sizes
            response_lens.append(len(ans.answer))
            
    print("\n==================================================")
    print("             LLM PIPELINE LATENCY PROFILE         ")
    print("==================================================")
    print(f"Iterations: {iterations}")
    print(f"Model Evaluated: {manager.model_name}")
    print("--------------------------------------------------")
    print(f"1. Prompt Construction Time:")
    print(f"   Average: {np.mean(prompt_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"2. Model Inference Latency:")
    print(f"   Average: {np.mean(inf_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"3. Formatting & Citation Compilation:")
    print(f"   Average: {np.mean(fmt_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"4. Total Response Pipeline Latency:")
    print(f"   Average: {np.mean(total_times):.2f} ms")
    print("--------------------------------------------------")
    print(f"5. Answer Payload size: {np.mean(response_lens):.1f} characters")
    print("==================================================\n")

if __name__ == "__main__":
    run_benchmark()
