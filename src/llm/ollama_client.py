import requests
from typing import Dict, Any, Tuple
from src.utils.logger import logger

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
        """Sends a prompt to Ollama generation endpoint with validations and diagnostics.
        
        Args:
            model_name: Target model tag.
            prompt: Structured text prompt.
            
        Returns:
            A tuple of (generated_response_text, token_statistics_dictionary, diagnostics_dictionary)
        """
        from src.config.settings import MAX_CONTEXT_CHARACTERS
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
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 300,
                "num_ctx": 2048
            }
        }
        
        # 2. Payload Validation
        allowed_keys = {"model", "prompt", "stream", "options"}
        if set(payload.keys()) != allowed_keys:
            raise ValueError(f"Payload validation failed: payload contains unexpected keys: {set(payload.keys()) - allowed_keys}")
            
        # Log the payload structure (excluding the full prompt)
        payload_log = payload.copy()
        payload_log["prompt"] = f"<PROMPT_PREVIEW: {len(prompt)} characters>"
        logger.info(f"Ollama API request payload: {payload_log}")

        # 3. Request Diagnostics Logging
        timestamp_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info("Ollama Request Diagnostics:")
        logger.info(f"- Timestamp: {timestamp_str}")
        logger.info(f"- Model name: {model_name}")
        logger.info(f"- Prompt length: {len(prompt)} characters")
        logger.info(f"- Configured MAX_CONTEXT_CHARACTERS: {MAX_CONTEXT_CHARACTERS}")
        logger.info("- Stream mode: False")
        logger.info(f"- Request options: {payload['options']}")
        logger.info(f"- Prompt preview (first 1000 characters): {prompt[:1000]}")

        # Initialize diagnostics output
        diagnostics = {
            "model": model_name,
            "prompt_length": len(prompt),
            "context_limit": MAX_CONTEXT_CHARACTERS,
            "request_time": timestamp_str,
            "inference_time_ms": 0.0,
            "http_status": None,
            "returned_characters": 0,
            "ollama_error": None
        }

        # 4. Generate with 180s Timeout & Diagnostic Tracking
        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=180.0)
            elapsed_time_ms = (time.time() - start_time) * 1000.0
            diagnostics["inference_time_ms"] = elapsed_time_ms
            diagnostics["http_status"] = response.status_code
        except Exception as conn_err:
            elapsed_time_ms = (time.time() - start_time) * 1000.0
            diagnostics["inference_time_ms"] = elapsed_time_ms
            diagnostics["ollama_error"] = str(conn_err)
            
            logger.error("Ollama Response Diagnostics [FAILED]:")
            logger.error(f"- HTTP status: Connection Error")
            logger.error(f"- Total request time: {elapsed_time_ms:.2f} ms")
            logger.error(f"- Returned response size: N/A")
            logger.error(f"- Ollama error body: {str(conn_err)}")
            raise conn_err

        # 5. Response Diagnostics Logging
        logger.info("Ollama Response Diagnostics:")
        logger.info(f"- HTTP status: {response.status_code}")
        logger.info(f"- Total request time: {elapsed_time_ms:.2f} ms")

        # 6. Detailed Error Processing
        if response.status_code != 200:
            try:
                error = response.json().get("error", response.text)
            except Exception:
                error = response.text
                
            diagnostics["ollama_error"] = error
            logger.error(f"- Returned response size: N/A")
            logger.error(f"- Ollama error body: {error}")
            raise RuntimeError(f"Ollama API {response.status_code}: {error}")

        data = response.json()
        answer = data.get("response", "").strip()
        diagnostics["returned_characters"] = len(answer)
        
        logger.info(f"- Returned response size: {len(answer)} characters")
        logger.info("- Ollama error body: None")
        
        stats = {
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0)
        }
        
        return answer, stats, diagnostics
