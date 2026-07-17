import requests
import json
import os
from typing import Dict, Any, Tuple
from src.utils.logger import logger

def write_runtime_log(diagnostics: Dict[str, Any], success: bool):
    try:
        from src.config.settings import LOGS_DIR
        log_file = os.path.join(LOGS_DIR, "ollama_runtime.jsonl")
        record = {
            "timestamp": diagnostics.get("request_time"),
            "runtime_mode": diagnostics.get("runtime_mode"),
            "runtime_profile": diagnostics.get("runtime_profile"),
            "num_ctx": diagnostics.get("context_limit"),
            "num_predict": diagnostics.get("num_predict"),
            "latency": diagnostics.get("inference_time_ms"),
            "retry": diagnostics.get("retry_performed"),
            "retry_reason": diagnostics.get("retry_reason"),
            "HTTP_status": diagnostics.get("http_status"),
            "success": success
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write to ollama_runtime.jsonl: {str(e)}")

class OllamaClient:
    """Interfaces with local offline Ollama API endpoints for model state validation and generation."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initializes the base URL.
        
        Args:
            base_url: Port address where Ollama is listening (default: http://localhost:11434).
        """
        self.base_url = base_url

    def is_server_running(self) -> bool:
        """Sends a GET request to verify Ollama service status.
        
        Returns:
            True if server is active.
        """
        try:
            response = requests.get(self.base_url, timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    def is_model_installed(self, model_name: str) -> bool:
        """Checks local tags registry list to confirm if target model is pulled.
        
        Args:
            model_name: Configured model identifier (e.g. phi3:mini).
            
        Returns:
            True if model is installed.
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=3.0)
            if response.status_code != 200:
                return False
                
            data = response.json()
            models = data.get("models", [])
            # Search matches by name or parameter variation (e.g. phi3:mini, phi3:latest)
            installed_names = [m.get("name") for m in models]
            
            # Match exact or base tags
            for name in installed_names:
                if name == model_name or name.split(":")[0] == model_name.split(":")[0]:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Ollama registry lookup failed: {str(e)}")
            return False

    def generate(self, model_name: str, prompt: str) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        """Sends a prompt to Ollama generation endpoint with validations, profile loading, and GPU/CPU fallback retries.
        
        Args:
            model_name: Target model tag.
            prompt: Structured text prompt.
            
        Returns:
            A tuple of (generated_response_text, token_statistics_dictionary, diagnostics_dictionary)
        """
        from src.config.settings import (
            MAX_CONTEXT_CHARACTERS, 
            LLM_RUNTIME_MODE, 
            LLM_RUNTIME_PROFILE, 
            OLLAMA_OPTIONS, 
            OLLAMA_PROFILES,
            AUTO_CPU_FALLBACK
        )
        import time
        
        # 1. Prompt Validation
        if prompt is None:
            raise ValueError("Prompt validation failed: prompt is None")
        if not isinstance(prompt, str):
            raise TypeError("Prompt validation failed: prompt must be a string")
        if not prompt.strip():
            raise ValueError("Prompt validation failed: prompt is empty or contains only whitespace")
        if len(prompt) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(f"Prompt validation failed: prompt length {len(prompt)} exceeds MAX_CONTEXT_CHARACTERS limit of {MAX_CONTEXT_CHARACTERS}")

        url = f"{self.base_url}/api/generate"
        
        # 2. Config options and CPU Mode translation
        options = OLLAMA_OPTIONS.copy()
        if LLM_RUNTIME_MODE == "cpu":
            options["num_gpu"] = 0
            
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        
        # Verify payload contains only expected keys
        allowed_keys = {"model", "prompt", "stream", "options"}
        if set(payload.keys()) != allowed_keys:
            raise ValueError(f"Payload validation failed: payload contains unexpected keys: {set(payload.keys()) - allowed_keys}")

        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # Initialize diagnostics
        diagnostics = {
            "model": model_name,
            "prompt_length": len(prompt),
            "context_limit": options.get("num_ctx", 1024),
            "num_predict": options.get("num_predict", 300),
            "runtime_mode": LLM_RUNTIME_MODE,
            "runtime_profile": LLM_RUNTIME_PROFILE,
            "request_time": timestamp_str,
            "inference_time_ms": 0.0,
            "http_status": None,
            "returned_characters": 0,
            "ollama_error": None,
            "gpu_oom_detected": "No",
            "retry_performed": "No",
            "retry_reason": None,
            "final_runtime_mode": "CPU" if options.get("num_gpu", -1) == 0 else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto"),
            "final_runtime_used": "CPU" if options.get("num_gpu", -1) == 0 else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto"),
            "cpu_fallback_triggered": False
        }

        def raise_runtime_error(msg: str):
            exc = RuntimeError(msg)
            exc.diagnostics = diagnostics
            raise exc

        def is_oom_error(err_msg: str) -> bool:
            oom_indicators = [
                "out of memory",
                "cudamalloc failed",
                "failed to allocate buffer",
                "failed to allocate kv cache"
            ]
            err_lower = err_msg.lower()
            return any(ind in err_lower for ind in oom_indicators)

        def make_request(current_payload):
            # Log the payload structure (excluding the full prompt)
            payload_log = current_payload.copy()
            payload_log["prompt"] = f"<PROMPT_PREVIEW: {len(prompt)} characters>"
            logger.info(f"Ollama API request payload: {payload_log}")

            logger.info("Ollama Request Diagnostics:")
            logger.info(f"- Timestamp: {timestamp_str}")
            logger.info(f"- Model name: {model_name}")
            logger.info(f"- Prompt length: {len(prompt)} characters")
            logger.info(f"- Configured num_ctx: {current_payload['options'].get('num_ctx')}")
            logger.info("- Stream mode: False")
            logger.info(f"- Request options: {current_payload['options']}")
            logger.info(f"- Prompt preview (first 1000 characters): {prompt[:1000]}")

            req_start = time.time()
            try:
                res = requests.post(url, json=current_payload, timeout=180.0)
                elapsed = (time.time() - req_start) * 1000.0
                return res, elapsed, None
            except Exception as e:
                elapsed = (time.time() - req_start) * 1000.0
                return None, elapsed, str(e)

        # 3. First Attempt
        response, elapsed_time_ms, conn_err = make_request(payload)
        diagnostics["inference_time_ms"] = elapsed_time_ms

        if conn_err:
            diagnostics["ollama_error"] = conn_err
            logger.error("Ollama Response Diagnostics [FAILED]:")
            logger.error(f"- HTTP status: Connection Error")
            logger.error(f"- Total request time: {elapsed_time_ms:.2f} ms")
            logger.error(f"- Ollama error body: {conn_err}")
            write_runtime_log(diagnostics, success=False)
            raise_runtime_error(f"Ollama API Connection Error: {conn_err}")

        diagnostics["http_status"] = response.status_code
        logger.info("Ollama Response Diagnostics:")
        logger.info(f"- HTTP status: {response.status_code}")
        logger.info(f"- Total request time: {elapsed_time_ms:.2f} ms")

        # Parse detailed error body if status is not 200
        error_msg = None
        if response.status_code != 200:
            try:
                error_msg = response.json().get("error", response.text)
            except Exception:
                error_msg = response.text

        # 4. Smart OOM Detection & GPU/CPU Fallback Retry Loop
        if error_msg and is_oom_error(error_msg):
            logger.warning(f"Ollama GPU OOM detected: '{error_msg}'. Retrying once using LOW_MEMORY profile...")
            diagnostics["gpu_oom_detected"] = "Yes"
            diagnostics["retry_performed"] = "Yes"
            diagnostics["retry_reason"] = "GPU OOM detected"
            
            low_options = OLLAMA_OPTIONS.copy()
            low_options.update(OLLAMA_PROFILES["LOW_MEMORY"])
            if LLM_RUNTIME_MODE == "cpu":
                low_options["num_gpu"] = 0
            
            payload["options"] = low_options
            diagnostics["context_limit"] = low_options.get("num_ctx", 512)
            diagnostics["num_predict"] = low_options.get("num_predict", 200)
            diagnostics["runtime_profile"] = "LOW_MEMORY"
            diagnostics["final_runtime_mode"] = "CPU" if low_options.get("num_gpu", -1) == 0 else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto")
            diagnostics["final_runtime_used"] = "CPU" if low_options.get("num_gpu", -1) == 0 else ("GPU" if LLM_RUNTIME_MODE == "gpu" else "auto")
            
            # Second attempt (GPU low context profile retry)
            response, elapsed_time_ms, conn_err = make_request(payload)
            diagnostics["inference_time_ms"] += elapsed_time_ms
            
            if conn_err:
                diagnostics["ollama_error"] = conn_err
                write_runtime_log(diagnostics, success=False)
                raise_runtime_error(f"Ollama API Connection Error on Retry: {conn_err}")
                
            diagnostics["http_status"] = response.status_code
            
            error_msg = None
            if response.status_code != 200:
                try:
                    error_msg = response.json().get("error", response.text)
                except Exception:
                    error_msg = response.text
            
            # 4b. GPU Low Memory retry failed, check if AUTO_CPU_FALLBACK is enabled
            if error_msg and is_oom_error(error_msg):
                if AUTO_CPU_FALLBACK:
                    logger.warning("LOW_MEMORY profile retry failed with VRAM OOM. Falling back to CPU mode...")
                    diagnostics["retry_reason"] = "GPU OOM detected, fallback to CPU"
                    
                    cpu_options = OLLAMA_OPTIONS.copy()
                    cpu_options.update(OLLAMA_PROFILES["LOW_MEMORY"])
                    cpu_options["num_gpu"] = 0  # Force CPU Mode
                    
                    payload["options"] = cpu_options
                    diagnostics["final_runtime_mode"] = "CPU"
                    diagnostics["final_runtime_used"] = "CPU"
                    
                    # Third attempt (CPU mode execution fallback)
                    response, elapsed_time_ms, conn_err = make_request(payload)
                    diagnostics["inference_time_ms"] += elapsed_time_ms
                    
                    if conn_err:
                        diagnostics["ollama_error"] = conn_err
                        write_runtime_log(diagnostics, success=False)
                        raise_runtime_error(f"Ollama API Connection Error on CPU Fallback: {conn_err}")
                        
                    diagnostics["http_status"] = response.status_code
                    
                    error_msg = None
                    if response.status_code != 200:
                        try:
                            error_msg = response.json().get("error", response.text)
                        except Exception:
                            error_msg = response.text
                            
                    if error_msg:
                        # CPU mode failed as well
                        diagnostics["ollama_error"] = error_msg
                        write_runtime_log(diagnostics, success=False)
                        raise_runtime_error(
                            "Model inference failed because the GPU does not have enough available memory.\n\n"
                            "Suggestions:\n"
                            "• Close GPU-intensive applications.\n"
                            "• Restart Ollama.\n"
                            "• Use the Low Memory runtime profile.\n"
                            "• Switch to CPU mode.\n"
                            "• Reduce context size."
                        )
                    else:
                        # CPU succeeded!
                        diagnostics["cpu_fallback_triggered"] = True
                        diagnostics["context_limit"] = cpu_options.get("num_ctx", 512)
                        diagnostics["num_predict"] = cpu_options.get("num_predict", 200)
                        diagnostics["runtime_profile"] = "LOW_MEMORY"
                else:
                    # CPU Fallback disabled, raise suggestions warning
                    diagnostics["ollama_error"] = error_msg
                    write_runtime_log(diagnostics, success=False)
                    raise_runtime_error(
                        "Model inference failed because the GPU does not have enough available memory.\n\n"
                        "Suggestions:\n"
                        "• Close GPU-intensive applications.\n"
                        "• Restart Ollama.\n"
                        "• Use the Low Memory runtime profile.\n"
                        "• Switch to CPU mode.\n"
                        "• Reduce context size."
                    )
            elif error_msg:
                # GPU Low Memory failed with non-OOM error
                diagnostics["ollama_error"] = error_msg
                write_runtime_log(diagnostics, success=False)
                raise_runtime_error(f"Ollama API {response.status_code}: {error_msg}")
            else:
                # GPU Low Memory succeeded!
                diagnostics["context_limit"] = low_options.get("num_ctx", 512)
                diagnostics["num_predict"] = low_options.get("num_predict", 200)
                diagnostics["runtime_profile"] = "LOW_MEMORY"
        elif error_msg:
            # First attempt failed with non-OOM error
            diagnostics["ollama_error"] = error_msg
            write_runtime_log(diagnostics, success=False)
            raise_runtime_error(f"Ollama API {response.status_code}: {error_msg}")

        # 5. Successful Execution Payload Processing
        data = response.json()
        answer = data.get("response", "").strip()
        diagnostics["returned_characters"] = len(answer)
        
        logger.info(f"- Returned response size: {len(answer)} characters")
        logger.info("- Ollama error body: None")
        
        stats = {
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0)
        }
        
        # Log successful run
        write_runtime_log(diagnostics, success=True)
        return answer, stats, diagnostics
