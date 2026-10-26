import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent import prepare_answer
from main import app
from router import classify_intent


class RouterTests(unittest.IsolatedAsyncioTestCase):
    def test_routes_and_keyword_boundaries(self):
        cases = {
            "What does A3i do?": "learn",
            "Tell me about the scheduling bot": "learn",
            "Can NeXtory summarize a book?": "learn",
            "What is NeXtory pricing?": "pricing",
            "Can I get a quote?": "pricing",
            "Does the demo cost anything?": "pricing",
            "What does a free trial cost?": "pricing",
            "Book a demo to discuss pricing": "demo",
            "Talk to sales": "demo",
            "Please connect me to sales to discuss meeting notes": "demo",
            "How do I contact the sales team?": "demo",
            "Can I see a live demonstration?": "demo",
            "Are there additional fees for integrations?": "pricing",
            "Can A3i reduce operating costs?": "learn",
            "Can A3i reduce operating costs, and what is the price?": "pricing",
            "I want a trial": "demo",
            "Schedule a meeting": "demo",
            "I want to book a product demonstration.": "demo",
            "Schedule a demo": "demo",
            "Book a product walkthrough": "demo",
            "What does a demonstration cost?": "pricing",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(classify_intent(message), expected)

    def test_product_selection_keeps_pricing_intent(self):
        history = [
            {"role": "user", "content": "What is the price?"},
            {"role": "assistant", "content": "Which product do you mean: NeXtory, Send3r?"},
        ]
        self.assertEqual(classify_intent("NeXtory", history), "pricing")

    async def test_demo_does_not_call_model_or_search(self):
        with patch('agent.call_llm', new=AsyncMock()) as model, patch('agent.search_a3i_web') as search:
            answer = await prepare_answer('Talk to sales', [])
        self.assertIn('Book a Meeting', answer)
        model.assert_not_awaited()
        search.invoke.assert_not_called()

    def test_stream_exposes_demo_intent(self):
        with TestClient(app) as client:
            response = client.post('/chat/stream', json={'message': 'Book a demo'})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(events[0]['intent'], 'demo')
        self.assertIn('Book a Meeting', events[1]['text'])

    def test_complete_response_exposes_pricing_intent(self):
        with patch('main.chat', new=AsyncMock(return_value='Contact A3i for a quote.')), TestClient(app) as client:
            response = client.post('/chat', json={'message': 'NeXtory pricing'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['intent'], 'pricing')
