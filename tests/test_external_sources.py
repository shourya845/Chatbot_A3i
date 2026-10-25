import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import external_sources as sources
from tools import search_a3i_web, search_cache


class ExternalSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = patch.object(sources, 'DB_PATH', Path(directory.name) / 'sources.sqlite3')
        database.start()
        self.addCleanup(database.stop)
        environment = patch.dict('os.environ', {'TAVILY_API_KEY': ''})
        environment.start()
        self.addCleanup(environment.stop)
        fallback = patch('tools._search_tavily', return_value={'results': []})
        fallback.start()
        self.addCleanup(fallback.stop)
        search_cache.clear()
        self.item = {
            'url': 'https://www.linkedin.com/company/a3i-accelerators',
            'title': 'A3i-Accel',
            'content': 'A3i-Accel headquarters: Chicago. Website https://a3i-accel.com/',
            'published_date': '2026-01-01',
        }
        self.page = {'url': 'https://a3i-accel.com/partials/about.html',
                     'content': '# Company\nA3i company information.'}

    def test_only_explicit_public_company_urls_are_accepted(self):
        for url in [
            'http://www.linkedin.com/company/a3i-accelerators',
            'https://www.linkedin.com/company/unrelated',
            'https://www.linkedin.com/in/someone',
            'https://www.linkedin.com.evil.test/company/a3i-accelerators',
            'https://www.linkedin.com@evil.test/company/a3i-accelerators',
            'https://en.wikipedia.org/wiki/Special:Search',
            'https://nvidia.com:9999/news',
            'https://nvidia.com:bad/news',
            'https://nvidia.com/news?redirect=http://localhost',
            'https://127.0.0.1/news',
        ]:
            with self.subTest(url=url):
                self.assertFalse(sources.allowed_url(url))
        self.assertTrue(sources.allowed_url(self.item['url']))
        self.assertTrue(sources.allowed_url('https://www.linkedin.com/posts/a3i-accelerators_news-123'))
        self.assertTrue(sources.allowed_url('https://en.wikipedia.org/wiki/A3i-Accel'))
        self.assertTrue(sources.allowed_url('https://blogs.nvidia.com/blog/example/'))

    def test_pending_snapshot_is_deduplicated_but_never_served(self):
        first = sources.store_candidates([{**self.item, 'status': 'approved'}])
        self.assertEqual(first, sources.store_candidates([self.item]))
        self.assertEqual(sources.list_sources()[0]['status'], 'pending_review')
        self.assertEqual(sources.approved_results('A3i headquarters'), [])

    def test_approval_preserves_dates_and_changed_text_needs_new_review(self):
        source_id = sources.store_candidates([self.item])[0]
        sources.review(source_id, 'approved')
        changed = {**self.item, 'content': self.item['content'].replace('Chicago', 'Elsewhere')}
        changed_id = sources.store_candidates([changed])[0]
        self.assertNotEqual(source_id, changed_id)
        results = sources.approved_results('A3i headquarters')
        self.assertEqual(len(results), 1)
        self.assertIn('Chicago', results[0]['content'])
        self.assertEqual(results[0]['published_at'], '2026-01-01')
        self.assertTrue(results[0]['retrieved_at'])
        self.assertTrue(results[0]['reviewed_at'])
        self.assertEqual(results[0]['source'], 'reviewed_external')
        sources.review(changed_id, 'approved')
        self.assertIn('Elsewhere', sources.approved_results('A3i headquarters')[0]['content'])

    def test_reject_does_not_get_reset_by_rediscovery(self):
        source_id = sources.store_candidates([self.item])[0]
        sources.review(source_id, 'rejected')
        sources.store_candidates([self.item])
        self.assertEqual(sources.list_sources()[0]['status'], 'rejected')
        with self.assertRaises(ValueError):
            sources.review(123456, 'approved')

    def test_third_party_candidates_must_identify_exact_company_and_domain(self):
        item = {**self.item, 'url': 'https://en.wikipedia.org/wiki/A3i-Accel'}
        self.assertEqual(sources.store_candidates([{**item, 'content': 'A different A3i organization.'}]), [])
        self.assertEqual(sources.store_candidates([{**item, 'content': 'A3i-Accel headquarters somewhere.'}]), [])
        self.assertTrue(sources.store_candidates([item]))

    def test_sensitive_and_product_queries_never_use_external_evidence(self):
        source_id = sources.store_candidates([self.item])[0]
        sources.review(source_id, 'approved')
        for question in ['A3i pricing news', 'A3i features and partnerships',
                         'A3i privacy policy news', 'NeXtory founder',
                         'A3i demo announcement', 'A3i headquarters and refund terms']:
            with self.subTest(question=question):
                self.assertEqual(sources.approved_results(question), [])
        with patch('tools.website_results', return_value=[self.page]), patch('tools.discover') as discover:
            search_a3i_web.invoke({'query': 'A3i pricing news'})
        discover.assert_not_called()

    def test_approval_and_revocation_take_effect_even_with_cached_official_search(self):
        source_id = sources.store_candidates([self.item])[0]
        with patch('tools.website_results', return_value=[self.page]) as reader:
            before = search_a3i_web.invoke({'query': 'A3i headquarters'})
            self.assertEqual(len(before['results']), 1)
            sources.review(source_id, 'approved')
            after = search_a3i_web.invoke({'query': 'A3i headquarters'})
            self.assertEqual(after['results'][0]['priority'], 'primary')
            self.assertEqual(after['results'][1]['priority'], 'secondary')
            sources.review(source_id, 'rejected')
            self.assertEqual(len(search_a3i_web.invoke({'query': 'A3i headquarters'})['results']), 1)
        reader.assert_called_once()

    def test_automatic_discovery_keeps_pending_content_out_of_model_results(self):
        def collect(query):
            return sources.store_candidates([self.item])
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'not-a-real-key'}), \
                patch('tools.website_results', return_value=[self.page]), \
                patch('tools.discover', side_effect=collect) as discover:
            for _ in range(2):
                result = search_a3i_web.invoke({'query': 'A3i headquarters'})
                self.assertNotIn('Chicago', str(result))
        discover.assert_called_once()
        self.assertEqual(sources.list_sources()[0]['status'], 'pending_review')

    def test_external_outage_does_not_break_official_answers(self):
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'not-a-real-key'}), \
                patch('tools.website_results', return_value=[self.page]), \
                patch('tools.discover', side_effect=RuntimeError('private provider detail')):
            result = search_a3i_web.invoke({'query': 'A3i headquarters'})
        self.assertNotIn('error', result)
        self.assertNotIn('private provider detail', str(result))

    def test_approved_background_is_available_when_official_retrieval_is_down(self):
        source_id = sources.store_candidates([self.item])[0]
        sources.review(source_id, 'approved')
        with patch('tools.website_results', return_value=[]):
            result = search_a3i_web.invoke({'query': 'A3i headquarters'})
        self.assertNotIn('error', result)
        self.assertEqual(result['results'][0]['url'], self.item['url'])

    def test_secondary_sources_do_not_expand_the_total_evidence_budget(self):
        source_id = sources.store_candidates([{**self.item, 'content': self.item['content'] * 150}])[0]
        sources.review(source_id, 'approved')
        pages = [{**self.page, 'content': 'Company background information. ' * 200}] * 2
        with patch('tools.website_results', return_value=pages):
            result = search_a3i_web.invoke({'query': 'A3i headquarters'})
        self.assertLessEqual(sum(len(row['content']) for row in result['results']), 7200)
        self.assertEqual(result['results'][-1]['priority'], 'secondary')

    def test_discovery_always_uses_server_controlled_domains(self):
        with patch('external_sources.TavilySearch') as search:
            search.return_value.invoke.return_value = {'results': [self.item]}
            self.assertTrue(sources.discover('A3i headquarters'))
        self.assertEqual(search.call_args.kwargs['include_domains'], sources.DISCOVERY_DOMAINS)
        self.assertEqual(search.return_value.invoke.call_args.args[0]['include_domains'], sources.DISCOVERY_DOMAINS)

    def test_approved_citation_survives_the_complete_agent_tool_workflow(self):
        from agent import chat
        source_id = sources.store_candidates([self.item])[0]
        sources.review(source_id, 'approved')
        question = 'Where is A3i headquartered?'
        replies = [
            {'role': 'assistant', 'tool_calls': [{'function': {
                'name': 'search_a3i_web', 'arguments': {'query': question}}}]},
            {'role': 'assistant', 'content': 'According to its LinkedIn profile, A3i is in '
             'Chicago. [LinkedIn](' + self.item['url'] + ')'},
        ]
        with patch('agent.call_llm', new=AsyncMock(side_effect=replies)), \
                patch('tools.website_results', return_value=[self.page]):
            answer = asyncio.run(chat(question, []))
        self.assertIn(self.item['url'], answer)
