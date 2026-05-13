"""
Tests for RAG retrieval.
Run with: python manage.py test chatbot.tests.test_retrieval
"""
from unittest.mock import MagicMock, patch

from django.test import TestCase

from chatbot.services import retrieval


class RetrievalTest(TestCase):

    def test_search_returns_list(self):
        mock_point = MagicMock()
        mock_point.score   = 0.8
        mock_point.payload = {"text": "Les vaches Holstein produisent beaucoup de lait."}

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        with patch.object(retrieval, "_get_qdrant") as mock_qdrant:
            mock_qdrant.return_value.query_points.return_value = mock_results
            with patch.object(retrieval, "_embed", return_value=[0.1] * 384):
                retrieval._initialized = True   # skip init
                results = retrieval.search("production laitière")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIn("Holstein", results[0])

    def test_low_score_filtered_out(self):
        mock_point = MagicMock()
        mock_point.score   = 0.2   # below threshold
        mock_point.payload = {"text": "Some irrelevant text"}

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        with patch.object(retrieval, "_get_qdrant") as mock_qdrant:
            mock_qdrant.return_value.query_points.return_value = mock_results
            with patch.object(retrieval, "_embed", return_value=[0.1] * 384):
                retrieval._initialized = True
                results = retrieval.search("anything")

        self.assertEqual(results, [])

    def test_search_returns_empty_on_no_results(self):
        mock_results = MagicMock(spec=[])   # no .points attribute
        with patch.object(retrieval, "_get_qdrant") as mock_qdrant:
            mock_qdrant.return_value.query_points.return_value = mock_results
            with patch.object(retrieval, "_embed", return_value=[0.1] * 384):
                retrieval._initialized = True
                results = retrieval.search("anything")
        self.assertEqual(results, [])