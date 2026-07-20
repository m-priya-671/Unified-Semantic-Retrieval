import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.llm.response_validator import ResponseValidator

class TestResponseValidator(unittest.TestCase):

    def setUp(self):
        self.context = (
            "FAISS is an open-source library developed by Meta for efficient similarity search "
            "and clustering of dense vectors. It was released in 2017. The system handles 1000000 vectors "
            "with low latency."
        )
        self.query = "What is FAISS?"

    def test_natural_paraphrased_answer_passes(self):
        answer = (
            "FAISS is a popular open-source vector search toolkit created by Meta. "
            "It helps perform fast similarity searches and clustering on high-dimensional dense embeddings."
        )
        is_valid, diag = ResponseValidator.validate_detailed(answer, self.context, self.query)
        self.assertTrue(is_valid)
        self.assertTrue(diag["validation_passed"])
        self.assertFalse(diag["validation_failed"])
        self.assertEqual(len(diag["unsupported_entities_found"]), 0)
        self.assertGreater(diag["semantic_similarity_score"], 0.40)

    def test_summary_generated_from_text_passes(self):
        summary_answer = (
            "Overall, FAISS is an efficient Meta library designed for clustering and vector similarity search. "
            "In summary, it manages dense vectors with high speed and low latency."
        )
        is_valid, diag = ResponseValidator.validate_detailed(summary_answer, self.context, "Summarize FAISS", intent="SUMMARY")
        self.assertTrue(is_valid)
        self.assertTrue(diag["validation_passed"])
        self.assertEqual(len(diag["unsupported_entities_found"]), 0)

    def test_connective_words_do_not_trigger_hallucination(self):
        answer = (
            "Furthermore, according to the document, FAISS is a library developed by Meta. "
            "However, it is specifically focused on similarity search and vector clustering."
        )
        is_valid, diag = ResponseValidator.validate_detailed(answer, self.context, self.query)
        self.assertTrue(is_valid)
        self.assertNotIn("Furthermore", diag["unsupported_entities_found"])
        self.assertNotIn("According", diag["unsupported_entities_found"])
        self.assertNotIn("However", diag["unsupported_entities_found"])

    def test_unsupported_entities_rejected(self):
        # Microsoft is not in context (Meta is in context)
        answer = "FAISS is a vector search library developed by Microsoft in California."
        is_valid, diag = ResponseValidator.validate_detailed(answer, self.context, self.query)
        self.assertFalse(is_valid)
        self.assertTrue(diag["validation_failed"])
        self.assertIn("Microsoft", diag["unsupported_entities_found"])

    def test_unsupported_numbers_rejected(self):
        # 5000000 is not in context (1000000 is in context)
        answer = "FAISS handles 5000000 dense vectors with low latency."
        is_valid, diag = ResponseValidator.validate_detailed(answer, self.context, self.query)
        self.assertFalse(is_valid)
        self.assertTrue(diag["validation_failed"])
        self.assertIn("5000000", diag["unsupported_entities_found"])

    def test_unsupported_dates_rejected(self):
        # 1999 is not in context (2017 is in context)
        answer = "FAISS was released in 1999 by Meta for similarity search."
        is_valid, diag = ResponseValidator.validate_detailed(answer, self.context, self.query)
        self.assertFalse(is_valid)
        self.assertTrue(diag["validation_failed"])
        self.assertIn("1999", diag["unsupported_entities_found"])

    def test_valid_decline_response_passes(self):
        decline_answer = "The uploaded document contains limited information related to your question, so a detailed answer cannot be generated from the available content."
        is_valid, diag = ResponseValidator.validate_detailed(decline_answer, self.context, "What is Quantum Computing?")
        self.assertTrue(is_valid)
        self.assertEqual(diag["validation_reason"], "Valid decline response")

if __name__ == "__main__":
    unittest.main()
