import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.retrieval.metadata import RetrievalChunk
from src.llm.prompt_builder import PromptBuilder

class TestPromptBuilder(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            RetrievalChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                source_file="doc1.pdf",
                source_reference="Page 1",
                similarity_score=0.95,
                confidence="HIGH",
                chunk_text="This is rank 1 context chunk with high priority details."
            ),
            RetrievalChunk(
                chunk_id="chunk_2",
                document_id="doc_2",
                source_file="doc2.pdf",
                source_reference="Page 2",
                similarity_score=0.85,
                confidence="MEDIUM",
                chunk_text="This is rank 2 context chunk with extra information."
            ),
            RetrievalChunk(
                chunk_id="chunk_3",
                document_id="doc_3",
                source_file="doc3.pdf",
                source_reference="Page 3",
                similarity_score=0.75,
                confidence="LOW",
                chunk_text="This is rank 3 context chunk with supplementary facts."
            )
        ]

    def test_small_prompt(self):
        query = "What is RAG?"
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=self.sample_chunks,
            max_context_chars=4000
        )
        self.assertLessEqual(len(prompt), 4000)
        self.assertEqual(count, 3)
        self.assertEqual(diag["chunks_trimmed"], 0)
        self.assertEqual(diag["chunks_included"], 3)
        self.assertEqual(diag["trim_reason"], "None")
        self.assertGreater(diag["available_context_budget"], 0)
        self.assertGreater(diag["context_utilization_percent"], 0.0)

    def test_prompt_exactly_at_limit(self):
        query = "Exact limit test"
        prompt_empty, _, _, diag_empty = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[],
            max_context_chars=4000
        )
        base_size = diag_empty["base_prompt_chars"]
        avail_budget = 4000 - base_size
        
        header_len = len("[Chunk 1]\nSource: doc1.pdf\nReference: Page 1\nSimilarity: 0.90\n\n")
        filler_text = "A" * (avail_budget - header_len)
        
        exact_chunk = RetrievalChunk(
            chunk_id="c_exact",
            document_id="d1",
            source_file="doc1.pdf",
            source_reference="Page 1",
            similarity_score=0.90,
            confidence="HIGH",
            chunk_text=filler_text
        )
        
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[exact_chunk],
            max_context_chars=4000
        )
        self.assertEqual(len(prompt), 4000)
        self.assertEqual(count, 1)
        self.assertEqual(diag["remaining_budget"], 0)
        self.assertEqual(diag["context_utilization_percent"], 100.0)

    def test_prompt_exceeding_limit(self):
        query = "Too large prompt"
        huge_chunk = RetrievalChunk(
            chunk_id="huge_1",
            document_id="d_huge",
            source_file="huge.pdf",
            source_reference="Page 1",
            similarity_score=0.99,
            confidence="HIGH",
            chunk_text="X" * 5000
        )
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[huge_chunk],
            max_context_chars=4000
        )
        self.assertLessEqual(len(prompt), 4000)
        self.assertEqual(count, 0)
        self.assertEqual(diag["chunks_trimmed"], 1)
        self.assertEqual(diag["trim_reason"], "Exceeded context budget")

    def test_automatic_trimming(self):
        query = "Automatic trimming test"
        chunks = [
            RetrievalChunk("c1", "d1", "f1.pdf", "P1", 0.9, "HIGH", "A" * 1500),
            RetrievalChunk("c2", "d2", "f2.pdf", "P2", 0.8, "HIGH", "B" * 1500),
            RetrievalChunk("c3", "d3", "f3.pdf", "P3", 0.7, "HIGH", "C" * 1500)
        ]
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=chunks,
            max_context_chars=4000
        )
        self.assertLessEqual(len(prompt), 4000)
        self.assertLess(count, 3)
        self.assertGreater(diag["chunks_trimmed"], 0)

    def test_chunk_preservation(self):
        query = "Preservation test"
        chunk_text = "DO_NOT_TRUNCATE_THIS_FULL_SENTENCE_TEXT"
        chunk = RetrievalChunk("c1", "d1", "f1.pdf", "P1", 0.9, "HIGH", chunk_text)
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[chunk],
            max_context_chars=4000
        )
        self.assertIn(chunk_text, context)
        self.assertTrue(context.endswith(chunk_text))

    def test_ranked_chunk_inclusion(self):
        query = "Ranked inclusion test"
        c1 = RetrievalChunk("c1", "d1", "f1.pdf", "P1", 0.99, "HIGH", "Rank 1 Content")
        c2 = RetrievalChunk("c2", "d2", "f2.pdf", "P2", 0.88, "MEDIUM", "Rank 2 Content")
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[c1, c2],
            max_context_chars=4000
        )
        self.assertEqual(count, 2)
        pos_rank1 = context.find("Rank 1 Content")
        pos_rank2 = context.find("Rank 2 Content")
        self.assertTrue(pos_rank1 < pos_rank2)

    def test_skip_and_continue_behavior(self):
        """
        Scenario:
        Rank 1 = fits
        Rank 2 = too large
        Rank 3 = fits
        Rank 4 = fits
        Verify:
        - Rank 2 is skipped.
        - Rank 3 and Rank 4 are included.
        - Final prompt remains within budget.
        - No chunk is truncated.
        """
        query = "Skip and continue test"
        max_limit = 1300
        prompt_empty, _, _, diag_empty = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[],
            max_context_chars=max_limit
        )
        base_size = diag_empty["base_prompt_chars"]

        r1 = RetrievalChunk("r1", "d1", "f1.pdf", "P1", 0.95, "HIGH", "Small rank 1 text.")
        r2 = RetrievalChunk("r2", "d2", "f2.pdf", "P2", 0.85, "MEDIUM", "X" * 450)
        r3 = RetrievalChunk("r3", "d3", "f3.pdf", "P3", 0.75, "LOW", "Small rank 3 text.")
        r4 = RetrievalChunk("r4", "d4", "f4.pdf", "P4", 0.65, "LOW", "R4 text.")

        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=[r1, r2, r3, r4],
            max_context_chars=max_limit
        )

        self.assertLessEqual(len(prompt), max_limit)
        self.assertIn("Small rank 1 text.", context)
        self.assertNotIn("X" * 450, context)  # Rank 2 skipped!
        self.assertIn("Small rank 3 text.", context)  # Rank 3 included!
        self.assertIn("R4 text.", context)  # Rank 4 included!
        self.assertEqual(diag["chunks_trimmed"], 1)

    def test_budget_utilization(self):
        query = "Utilization test"
        chunks = [
            RetrievalChunk("c1", "d1", "f1.pdf", "P1", 0.9, "HIGH", "Text 1"),
            RetrievalChunk("c2", "d2", "f2.pdf", "P2", 0.8, "HIGH", "Text 2")
        ]
        prompt, context, count, diag = PromptBuilder.build_with_diagnostics(
            query=query,
            chunks=chunks,
            max_context_chars=4000
        )
        self.assertGreater(diag["context_utilization_percent"], 0.0)
        self.assertLessEqual(diag["context_utilization_percent"], 100.0)

    def test_defensive_validation(self):
        query = "Defensive validation test"
        chunks = [
            RetrievalChunk("c1", "d1", "f1.pdf", "P1", 0.9, "HIGH", "A" * 1000)
        ]
        prompt, context, count = PromptBuilder.build(
            query=query,
            chunks=chunks,
            max_context_chars=4000
        )
        self.assertLessEqual(len(prompt), 4000)

if __name__ == "__main__":
    unittest.main()
