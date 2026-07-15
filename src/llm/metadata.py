from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class GroundedAnswer:
    """Type-safe container holding grounded answer generations, source citations, and telemetry metrics."""
    success: bool
    reason: str  # "SUCCESS", "NO_RELEVANT_CONTEXT", "VALIDATION_FAILED", "OLLAMA_UNAVAILABLE", "MODEL_MISSING"
    message: str
    answer: str
    sources: List[str] = field(default_factory=list)
    conversation_id: str = "default_session"
    question_id: str = ""
    timestamp: str = ""
    prompt_version: str = "1.0"
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    token_statistics: Dict[str, int] = field(default_factory=dict)
