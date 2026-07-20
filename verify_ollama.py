import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import unittest
from unittest.mock import patch, MagicMock
import requests
import sys
from pathlib import Path

# Add workspace directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.llm.ollama_client import OllamaClient
from src.config.settings import (
    MAX_CONTEXT_CHARACTERS, 
    LLM_RUNTIME_MODE, 
    LLM_RUNTIME_PROFILE, 
    OLLAMA_OPTIONS, 
    OLLAMA_PROFILES
)

class TestOllamaClient(unittest.TestCase):
    
    def setUp(self):
        self.client = OllamaClient("http://localhost:11434")

    @patch("requests.get")
    def test_api_reachable(self, mock_get):
        # Setup mock active server response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        self.assertTrue(self.client.is_server_running())
        mock_get.assert_called_with("http://localhost:11434", timeout=3.0)

    @patch("requests.get")
    def test_model_available(self, mock_get):
        # Setup model tags mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "phi3:mini"}, {"name": "llama3:8b"}]
        }
        mock_get.return_value = mock_response
        
        self.assertTrue(self.client.is_model_installed("phi3:mini"))
        self.assertTrue(self.client.is_model_installed("phi3:latest"))
        self.assertFalse(self.client.is_model_installed("uninstalled-model"))

    @patch("requests.post")
    def test_small_prompt_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Small response",
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        mock_post.return_value = mock_response
        
        prompt = "Hello"
        answer, stats, diag = self.client.generate("phi3:mini", prompt)
        
        self.assertEqual(answer, "Small response")
        self.assertEqual(stats["eval_count"], 5)
        self.assertEqual(diag["prompt_length"], len(prompt))
        self.assertEqual(diag["returned_characters"], len(answer))
        self.assertIsNone(diag["ollama_error"])
        self.assertEqual(diag["http_status"], 200)

    @patch("requests.post")
    def test_medium_prompt_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Medium response answer",
            "prompt_eval_count": 100,
            "eval_count": 40
        }
        mock_post.return_value = mock_response
        
        prompt = "Explain RAG systems in 50 words."
        answer, stats, diag = self.client.generate("phi3:mini", prompt)
        
        self.assertEqual(answer, "Medium response answer")
        self.assertEqual(diag["prompt_length"], len(prompt))

    @patch("requests.post")
    def test_large_prompt_within_limit(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Large answer payload response.",
            "prompt_eval_count": 800,
            "eval_count": 150
        }
        mock_post.return_value = mock_response
        
        # Build prompt exactly at max context boundary
        prompt = "A" * MAX_CONTEXT_CHARACTERS
        answer, stats, diag = self.client.generate("phi3:mini", prompt)
        self.assertEqual(diag["prompt_length"], MAX_CONTEXT_CHARACTERS)

    @patch("requests.post")
    def test_prompt_exceeding_limit_defensive_trimming(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Trimmed prompt response.",
            "prompt_eval_count": 1000,
            "eval_count": 20
        }
        mock_post.return_value = mock_response
        
        # Exceeding context size should apply defensive trimming without raising ValueError
        too_long_prompt = "A" * (MAX_CONTEXT_CHARACTERS + 500)
        answer, stats, diag = self.client.generate("phi3:mini", too_long_prompt)
        self.assertEqual(answer, "Trimmed prompt response.")
        self.assertLessEqual(diag["prompt_length"], MAX_CONTEXT_CHARACTERS)

    def test_invalid_prompt_validations(self):
        # Empty string
        with self.assertRaises(ValueError) as ctx:
            self.client.generate("phi3:mini", "")
        self.assertIn("prompt is empty", str(ctx.exception))
        
        # Whitespace string
        with self.assertRaises(ValueError) as ctx2:
            self.client.generate("phi3:mini", "   ")
        self.assertIn("prompt is empty", str(ctx2.exception))
        
        # None prompt
        with self.assertRaises(ValueError) as ctx3:
            self.client.generate("phi3:mini", None)
        self.assertIn("prompt is None", str(ctx3.exception))

    @patch("requests.post")
    def test_timeout_handling(self, mock_post):
        # Simulate connection/read timeout
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out after 180s")
        
        with self.assertRaises(RuntimeError):
            self.client.generate("phi3:mini", "Trigger timeout query")
            
    @patch("requests.post")
    def test_error_reporting_detailed(self, mock_post):
        # Simulate HTTP 500 error with detailed JSON error payload
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "Generic database failure or network exception."
        }
        mock_post.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as ctx:
            self.client.generate("phi3:mini", "Trigger server error")
            
        self.assertIn("Ollama API 500: Generic database failure", str(ctx.exception))
        
        # Simulate HTTP 404 error with plain text error payload
        mock_response_txt = MagicMock()
        mock_response_txt.status_code = 404
        mock_response_txt.json.side_effect = Exception("Not JSON")
        mock_response_txt.text = "Model not found"
        mock_post.return_value = mock_response_txt
        
        with self.assertRaises(RuntimeError) as ctx2:
            self.client.generate("phi3:mini", "Trigger 404")
            
        self.assertIn("Ollama API 404: Model not found", str(ctx2.exception))

    def test_runtime_profile_loading(self):
        # Verify default settings configs
        self.assertIn(LLM_RUNTIME_PROFILE, OLLAMA_PROFILES)
        self.assertEqual(OLLAMA_OPTIONS["num_ctx"], OLLAMA_PROFILES[LLM_RUNTIME_PROFILE]["num_ctx"])
        self.assertEqual(OLLAMA_OPTIONS["num_predict"], OLLAMA_PROFILES[LLM_RUNTIME_PROFILE]["num_predict"])

    @patch("requests.post")
    @patch("src.config.settings.LLM_RUNTIME_MODE", "cpu")
    def test_cpu_mode_override(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "CPU result"}
        mock_post.return_value = mock_response
        
        answer, stats, diag = self.client.generate("phi3:mini", "Test query")
        # Assert post was called with options having num_gpu=0
        called_args, called_kwargs = mock_post.call_args
        self.assertEqual(called_kwargs["json"]["options"]["num_gpu"], 0)

    @patch("requests.post")
    def test_gpu_oom_detection_and_retry(self, mock_post):
        # Setup first attempt failing with OOM, second succeeding
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.json.return_value = {"error": "cudaMalloc failed: out of memory during initialization"}
        
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"response": "Recovered response"}
        
        mock_post.side_effect = [mock_fail, mock_success]
        
        answer, stats, diag = self.client.generate("phi3:mini", "OOM recovery query")
        self.assertEqual(answer, "Recovered response")
        self.assertEqual(diag["gpu_oom_detected"], "Yes")
        self.assertEqual(diag["retry_performed"], "Yes")
        self.assertEqual(diag["runtime_profile"], "LOW_MEMORY")
        self.assertEqual(diag["context_limit"], 512)

    @patch("requests.post")
    def test_gpu_oom_double_failure_friendly_message(self, mock_post):
        # Setup first attempt failing with OOM, second retry also failing with OOM
        mock_fail1 = MagicMock()
        mock_fail1.status_code = 500
        mock_fail1.json.return_value = {"error": "failed to allocate buffer for kv cache"}
        
        mock_fail2 = MagicMock()
        mock_fail2.status_code = 500
        mock_fail2.json.return_value = {"error": "cudaMalloc failed: out of memory"}
        
        mock_post.side_effect = [mock_fail1, mock_fail2]
        
        with self.assertRaises(RuntimeError) as ctx:
            self.client.generate("phi3:mini", "Double OOM trigger query")
            
        self.assertIn("GPU does not have enough available memory", str(ctx.exception))
        self.assertIn("Switch to CPU mode", str(ctx.exception))

    def test_all_profiles_exist(self):
        # Verify LOW_MEMORY, BALANCED, and QUALITY profiles exist and have correct fields
        self.assertIn("LOW_MEMORY", OLLAMA_PROFILES)
        self.assertIn("BALANCED", OLLAMA_PROFILES)
        self.assertIn("QUALITY", OLLAMA_PROFILES)
        
        self.assertEqual(OLLAMA_PROFILES["LOW_MEMORY"]["num_ctx"], 512)
        self.assertEqual(OLLAMA_PROFILES["LOW_MEMORY"]["num_predict"], 200)
        self.assertEqual(OLLAMA_PROFILES["BALANCED"]["num_ctx"], 1024)
        self.assertEqual(OLLAMA_PROFILES["BALANCED"]["num_predict"], 300)
        self.assertEqual(OLLAMA_PROFILES["QUALITY"]["num_ctx"], 2048)
        self.assertEqual(OLLAMA_PROFILES["QUALITY"]["num_predict"], 500)

    @patch("requests.post")
    @patch("src.config.settings.AUTO_CPU_FALLBACK", True)
    def test_cpu_fallback_success(self, mock_post):
        # 1st fail: GPU VRAM OOM
        mock_fail1 = MagicMock()
        mock_fail1.status_code = 500
        mock_fail1.json.return_value = {"error": "cudaMalloc failed: out of memory"}
        
        # 2nd fail: GPU LOW_MEMORY VRAM OOM
        mock_fail2 = MagicMock()
        mock_fail2.status_code = 500
        mock_fail2.json.return_value = {"error": "failed to allocate buffer for kv cache"}
        
        # 3rd success: CPU LOW_MEMORY success
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"response": "CPU fallback response answer"}
        
        mock_post.side_effect = [mock_fail1, mock_fail2, mock_success]
        
        answer, stats, diag = self.client.generate("phi3:mini", "CPU fallback search query")
        self.assertEqual(answer, "CPU fallback response answer")
        self.assertEqual(diag["cpu_fallback_triggered"], True)
        self.assertEqual(diag["final_runtime_used"], "CPU")
        self.assertEqual(diag["gpu_oom_detected"], "Yes")
        self.assertEqual(diag["retry_performed"], "Yes")
        self.assertEqual(diag["retry_reason"], "GPU OOM detected, fallback to CPU")

    def test_runtime_diagnostics_logging(self):
        # Check that runtime log file exists or gets created
        from src.config.settings import LOGS_DIR
        import os
        import json
        log_file = os.path.join(LOGS_DIR, "ollama_runtime.jsonl")
        
        # Remove if exists to check clean creation
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception:
                pass
                
        # Simulate writing a log
        test_diag = {
            "request_time": "2026-07-17T22:15:00Z",
            "runtime_mode": "auto",
            "runtime_profile": "BALANCED",
            "context_limit": 1024,
            "num_predict": 300,
            "inference_time_ms": 150.0,
            "retry_performed": "Yes",
            "retry_reason": "GPU OOM detected",
            "http_status": 200
        }
        
        from src.llm.ollama_client import write_runtime_log
        write_runtime_log(test_diag, success=True)
        
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["timestamp"], "2026-07-17T22:15:00Z")
        self.assertEqual(record["success"], True)

if __name__ == "__main__":
    unittest.main()
