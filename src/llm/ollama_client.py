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

    def generate(self, model_name: str, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Sends a prompt to Ollama generation endpoint.
        
        Args:
            model_name: Target model tag.
            prompt: Structured text prompt.
            
        Returns:
            A tuple of (generated_response_text, token_statistics_dictionary)
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Zero temperature for deterministic grounded extraction
                "num_predict": 300   # Limits maximum generation bounds to keep context compact
            }
        }
        
        logger.info(f"Submitting query generation to model '{model_name}'...")
        response = requests.post(url, json=payload, timeout=60.0)
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API request failed with status code: {response.status_code}")
            
        data = response.json()
        answer = data.get("response", "").strip()
        
        # Pull performance token counts
        stats = {
            "prompt_eval_count": data.get("prompt_eval_count", 0),  # Input tokens count
            "eval_count": data.get("eval_count", 0)                 # Output tokens count
        }
        
        logger.info(f"Ollama generation complete. Output tokens: {stats['eval_count']}")
        return answer, stats
