import time
import uuid
from typing import Dict, Any, List
from src.retrieval.metadata import RetrievalResult
from src.config.settings import (
    LLM_MODEL_NAME, 
    PROMPT_VERSION, 
    MAX_CONTEXT_CHARACTERS, 
    LLM_RUNTIME_MODE, 
    LLM_RUNTIME_PROFILE, 
    OLLAMA_OPTIONS
)
from src.llm.metadata import GroundedAnswer
from src.llm.prompt_builder import PromptBuilder
from src.llm.ollama_client import OllamaClient
from src.llm.response_validator import ResponseValidator
from src.llm.citation_formatter import CitationFormatter
from src.llm.answer_formatter import AnswerFormatter
from src.utils.logger import logger

class LLMManager:
    """Coordinates local LLM generation including service pre-checks, validations, and citation formatting."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initializes dependencies.
        
        Args:
            ollama_url: Base connection endpoint string.
        """
        self.client = OllamaClient(ollama_url)
        self.model_name = LLM_MODEL_NAME

    def generate_grounded_answer(
        self, 
        retrieval_result: RetrievalResult, 
        conversation_id: str = "default_session"
    ) -> GroundedAnswer:
        """Processes the semantic search output and generates a grounded response.
        
        Args:
            retrieval_result: Output RetrievalResult dataclass from RetrievalManager.
            conversation_id: Unique chat session string.
            
        Returns:
            A GroundedAnswer dataclass payload.
        """
        start_total = time.time()
        question_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # 1. No-Context Handling
        if not retrieval_result.success:
            logger.info("Skipping model inference: RetrievalResult reports failure.")
            latency_metrics = {
                "prompt_construction_time_ms": 0.0,
                "inference_time_ms": 0.0,
                "formatting_time_ms": 0.0,
                "total_response_latency_ms": (time.time() - start_total) * 1000.0
            }
            return GroundedAnswer(
                success=False,
                reason="NO_RELEVANT_CONTEXT",
                message="No relevant information matching your query was found in the indexed documents. Please upload documents related to this topic.",
                answer="No relevant information matching your query was found in the indexed documents. Please upload documents related to this topic.",
                conversation_id=conversation_id,
                question_id=question_id,
                timestamp=timestamp,
                prompt_version=PROMPT_VERSION,
                latency_metrics=latency_metrics,
                diagnostics={
                    "model": self.model_name,
                    "prompt_length": 0,
                    "context_length": 0,
                    "context_limit": OLLAMA_OPTIONS.get("num_ctx", 1024),
                    "num_predict": OLLAMA_OPTIONS.get("num_predict", 300),
                    "runtime_mode": LLM_RUNTIME_MODE,
                    "runtime_profile": LLM_RUNTIME_PROFILE,
                    "request_time": timestamp,
                    "inference_time_ms": 0.0,
                    "http_status": None,
                    "returned_characters": 0,
                    "ollama_error": "No relevant context found during retrieval.",
                    "gpu_oom_detected": "No",
                    "retry_performed": "No",
                    "final_runtime_mode": "CPU" if LLM_RUNTIME_MODE == "cpu" else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto")
                }
            )
            
        # 2. Pre-Inference Service Check
        if not self.client.is_server_running():
            msg = f"Ollama server is not running on {self.client.base_url}. Please start Ollama before searching."
            logger.error(msg)
            return GroundedAnswer(
                success=False,
                reason="OLLAMA_UNAVAILABLE",
                message=msg,
                answer=msg,
                conversation_id=conversation_id,
                question_id=question_id,
                timestamp=timestamp,
                prompt_version=PROMPT_VERSION,
                latency_metrics={"total_response_latency_ms": (time.time() - start_total) * 1000.0},
                diagnostics={
                    "model": self.model_name,
                    "prompt_length": 0,
                    "context_length": 0,
                    "context_limit": OLLAMA_OPTIONS.get("num_ctx", 1024),
                    "num_predict": OLLAMA_OPTIONS.get("num_predict", 300),
                    "runtime_mode": LLM_RUNTIME_MODE,
                    "runtime_profile": LLM_RUNTIME_PROFILE,
                    "request_time": timestamp,
                    "inference_time_ms": 0.0,
                    "http_status": None,
                    "returned_characters": 0,
                    "ollama_error": "Ollama server is not running.",
                    "gpu_oom_detected": "No",
                    "retry_performed": "No",
                    "final_runtime_mode": "CPU" if LLM_RUNTIME_MODE == "cpu" else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto")
                }
            )
            
        # 3. Model Registry Check
        if not self.client.is_model_installed(self.model_name):
            msg = f"Model '{self.model_name}' is not installed in your local Ollama registry. Please run 'ollama pull {self.model_name}' first."
            logger.error(msg)
            return GroundedAnswer(
                success=False,
                reason="MODEL_MISSING",
                message=msg,
                answer=msg,
                conversation_id=conversation_id,
                question_id=question_id,
                timestamp=timestamp,
                prompt_version=PROMPT_VERSION,
                latency_metrics={"total_response_latency_ms": (time.time() - start_total) * 1000.0},
                diagnostics={
                    "model": self.model_name,
                    "prompt_length": 0,
                    "context_length": 0,
                    "context_limit": OLLAMA_OPTIONS.get("num_ctx", 1024),
                    "num_predict": OLLAMA_OPTIONS.get("num_predict", 300),
                    "runtime_mode": LLM_RUNTIME_MODE,
                    "runtime_profile": LLM_RUNTIME_PROFILE,
                    "request_time": timestamp,
                    "inference_time_ms": 0.0,
                    "http_status": None,
                    "returned_characters": 0,
                    "ollama_error": f"Model {self.model_name} is missing.",
                    "gpu_oom_detected": "No",
                    "retry_performed": "No",
                    "final_runtime_mode": "CPU" if LLM_RUNTIME_MODE == "cpu" else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto")
                }
            )
 
        # 4. Prompt construction & context length check
        start_prompt = time.time()
        prompt, limited_context, chunks_used = PromptBuilder.build(
            query=retrieval_result.query,
            chunks=retrieval_result.retrieved_chunks,
            max_context_chars=MAX_CONTEXT_CHARACTERS
        )
        prompt_time = (time.time() - start_prompt) * 1000.0
 
        # 5. Local Model Query Inference
        diagnostics = {
            "model": self.model_name,
            "prompt_length": len(prompt),
            "context_length": len(limited_context),
            "context_limit": OLLAMA_OPTIONS.get("num_ctx", 1024),
            "num_predict": OLLAMA_OPTIONS.get("num_predict", 300),
            "runtime_mode": LLM_RUNTIME_MODE,
            "runtime_profile": LLM_RUNTIME_PROFILE,
            "request_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inference_time_ms": 0.0,
            "http_status": None,
            "returned_characters": 0,
            "ollama_error": None,
            "gpu_oom_detected": "No",
            "retry_performed": "No",
            "retry_reason": None,
            "final_runtime_mode": "CPU" if LLM_RUNTIME_MODE == "cpu" else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto"),
            "final_runtime_used": "CPU" if LLM_RUNTIME_MODE == "cpu" else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto"),
            "cpu_fallback_triggered": False
        }
        
        try:
            start_inf = time.time()
            raw_answer, token_stats, client_diag = self.client.generate(self.model_name, prompt)
            inference_time = (time.time() - start_inf) * 1000.0
            diagnostics.update(client_diag)
        except Exception as e:
            inference_time = (time.time() - start_inf) * 1000.0
            msg = str(e)
            logger.error(f"Model inference call failed: {msg}")
            
            ollama_error = str(e)
            diagnostics["inference_time_ms"] = inference_time
            diagnostics["ollama_error"] = ollama_error
            
            if hasattr(e, "diagnostics") and isinstance(e.diagnostics, dict):
                diagnostics.update(e.diagnostics)
            
            if "Ollama API" in ollama_error:
                try:
                    parts = ollama_error.split(" ")
                    if len(parts) >= 3:
                        diagnostics["http_status"] = int(parts[2].replace(":", ""))
                except Exception:
                    pass
            
            return GroundedAnswer(
                success=False,
                reason="INFERENCE_ERROR",
                message=msg,
                answer=msg,
                conversation_id=conversation_id,
                question_id=question_id,
                timestamp=timestamp,
                prompt_version=PROMPT_VERSION,
                latency_metrics={"total_response_latency_ms": (time.time() - start_total) * 1000.0},
                diagnostics=diagnostics
            )
 
        # 6. Response Quality & Entity Validation
        is_valid = ResponseValidator.validate(raw_answer, limited_context, retrieval_result.query)
        if not is_valid:
            msg = "I am sorry, but the model generated a response that could not be validated against the retrieved documents. Please verify your query or upload more context."
            logger.warning("Generation discarded due to validation checks failure.")
            latency_metrics = {
                "prompt_construction_time_ms": prompt_time,
                "inference_time_ms": inference_time,
                "formatting_time_ms": 0.0,
                "total_response_latency_ms": (time.time() - start_total) * 1000.0
            }
            diagnostics["ollama_error"] = "Grounded response validation failed (hallucinated entity or empty output)."
            return GroundedAnswer(
                success=False,
                reason="VALIDATION_FAILED",
                message=msg,
                answer=msg,
                conversation_id=conversation_id,
                question_id=question_id,
                timestamp=timestamp,
                prompt_version=PROMPT_VERSION,
                latency_metrics=latency_metrics,
                diagnostics=diagnostics
            )
 
        # 7. Citations & Formatting
        start_fmt = time.time()
        citations_md, sources_list = CitationFormatter.format(retrieval_result.retrieved_chunks[:chunks_used])
        formatted_answer = AnswerFormatter.format(raw_answer, citations_md)
        fmt_time = (time.time() - start_fmt) * 1000.0
        
        latency_metrics = {
            "prompt_construction_time_ms": prompt_time,
            "inference_time_ms": inference_time,
            "formatting_time_ms": fmt_time,
            "total_response_latency_ms": (time.time() - start_total) * 1000.0
        }
        
        logger.info(f"Grounded answer generation succeeded. Total Latency: {latency_metrics['total_response_latency_ms']:.2f}ms")
        return GroundedAnswer(
            success=True,
            reason="SUCCESS",
            message="Answer generated and validated successfully.",
            answer=formatted_answer,
            sources=sources_list,
            conversation_id=conversation_id,
            question_id=question_id,
            timestamp=timestamp,
            prompt_version=PROMPT_VERSION,
            is_low_confidence=retrieval_result.is_low_confidence,
            latency_metrics=latency_metrics,
            token_statistics=token_stats,
            diagnostics=diagnostics
        )
