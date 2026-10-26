import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk

import agent
import chat_cli
from conversation import resolve_followup
from router import classify_intent


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    def test_ollama_reasoning_matches_model_family(self):
        for name, expected in (('gpt-oss:20b', 'low'), ('qwen3:4b', False)):
            with self.subTest(model=name), patch.object(agent, 'MODEL', name):
                agent.provider_model.cache_clear()
                try:
                    model = agent.provider_model('ollama')
                    self.assertEqual(model.reasoning, expected)
                    self.assertEqual(model.num_predict, 2048)
                finally:
                    agent.provider_model.cache_clear()

    async def test_demo_variants_return_booking_link_without_model(self):
        for question in ('I want to book a product demonstration.', 'Schedule a demo',
                         'Book a product walkthrough'):
            with self.subTest(question=question), patch('agent.call_llm') as model:
                self.assertEqual(classify_intent(question), 'demo')
                self.assertEqual(await agent.prepare_answer(question, []), agent.DEMO_REPLY)
                model.assert_not_called()

    async def test_empty_reply_tries_next_provider(self):
        primary = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='')))
        local = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='Local answer')))
        with patch('agent.model_candidates', return_value=iter([('groq', primary), ('ollama', local)])):
            self.assertEqual((await agent.call_llm([]))['content'], 'Local answer')

    async def test_empty_stream_tries_next_provider(self):
        async def empty(messages):
            yield AIMessageChunk(content='')
        async def answer(messages):
            yield AIMessageChunk(content='Fallback answer')
        with patch('agent.model_candidates', return_value=iter([
            ('groq', SimpleNamespace(astream=empty)), ('ollama', SimpleNamespace(astream=answer)),
        ])):
            self.assertEqual([part async for part in agent.model_tokens([])], ['Fallback answer'])

    async def test_truncated_reply_tries_next_provider(self):
        primary = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(
            content='Unfinished table', response_metadata={'finish_reason': 'length'})))
        local = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content='Complete answer')))
        with patch('agent.model_candidates', return_value=iter([('groq', primary), ('ollama', local)])):
            self.assertEqual((await agent.call_llm([]))['content'], 'Complete answer')

    async def test_truncated_stream_is_not_reported_as_complete(self):
        async def truncated(messages):
            yield AIMessageChunk(content='Partial table')
            yield AIMessageChunk(content='', response_metadata={'done_reason': 'length'})
        with patch('agent.model_candidates', return_value=iter([('ollama', SimpleNamespace(astream=truncated))])):
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                _ = [part async for part in agent.model_tokens([])]

    async def test_both_providers_fail_without_saving_a_false_answer(self):
        failing = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError('private secret')))
        with patch('agent.model_candidates', return_value=iter([('groq', failing), ('ollama', failing)])):
            with self.assertRaisesRegex(ValueError, 'No model provider'):
                await agent.call_llm([])

    def test_diagnostics_identify_failure_without_provider_secrets(self):
        with self.assertLogs('agent', level='WARNING') as logs:
            agent.log_provider_failure('ollama', 'stream', ValueError('Empty model response: private secret'))
        output = '\n'.join(logs.output)
        self.assertIn('empty model response', output)
        self.assertIn('model=', output)
        self.assertNotIn('private secret', output)

    async def test_cli_failure_is_friendly_and_not_saved(self):
        async def broken(*args):
            raise ValueError('No model provider is available')
            yield
        with patch('builtins.input', side_effect=['What is A3i?', 'bye']), \
                patch('builtins.print') as output, patch('chat_cli.stream_chat', side_effect=broken), \
                patch('chat_cli.save_history') as save:
            await chat_cli.main()
        text = ' '.join(str(call) for call in output.call_args_list)
        self.assertIn('Please try again shortly', text)
        self.assertNotIn('search configuration', text)
        save.assert_not_called()

    def test_new_product_keeps_previous_pricing_context(self):
        history = [{'role': 'user', 'content': 'What is the pricing for NeXtory?'},
                   {'role': 'assistant', 'content': 'NeXtory links to Notes AI plans.'}]
        self.assertEqual(resolve_followup('What about SpeedRec?', history),
                         'SpeedRec pricing: What about SpeedRec?')
        self.assertEqual(classify_intent('What about SpeedRec?', history), 'pricing')

    async def test_catalog_lists_every_maintained_name_without_a_model(self):
        catalog = json.loads(Path(agent.__file__).with_name('product_catalog.json').read_text())
        with patch('agent.call_llm') as model:
            answer = await agent.prepare_answer('List all A3i products.', [])
        for product in catalog['products']:
            self.assertIn(f'[{product["name"]}]({product["url"]})', answer)
        self.assertIn('does not establish an alias', answer)
        model.assert_not_called()

    async def test_missing_catalog_falls_back_to_website_flow(self):
        reply = {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_a3i_web', 'arguments': {'query': 'List all A3i products.'}}}]}
        with patch('agent.Path.read_text', side_effect=OSError), \
                patch('agent.call_llm', new=AsyncMock(return_value=reply)), \
                patch('agent.search_a3i_web') as search:
            search.invoke.return_value = {'results': []}
            self.assertEqual(await agent.prepare_answer('List all A3i products.', []), agent.FALLBACK)
