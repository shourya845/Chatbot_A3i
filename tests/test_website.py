import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from tools import relevant_excerpt, search_a3i_web, search_cache
from website import PRODUCT_URLS, PageText, search_terms, website_results


class WebsiteTests(unittest.TestCase):
    def setUp(self):
        search_cache.clear()

    def test_leadership_roles_are_distinct_search_terms(self):
        self.assertEqual(search_terms('CEO COO co-founder founders'), {'ceo', 'coo', 'founder'})

    def test_pricing_overview_reads_every_catalog_product_page(self):
        catalog = json.loads((Path(__file__).resolve().parent.parent / 'website_sources.json').read_text())
        expected = {page['id'] for page in catalog if page.get('product') or page['id'] == 'home'}
        self.assertGreater(len(expected), 2)
        response = httpx.Response(200, text='<h2>Pricing</h2><p>Call for Pricing</p>',
                                  request=httpx.Request('GET', 'https://a3i-accel.com/'))
        with patch('website.httpx.Client.get', return_value=response) as get:
            results = website_results('Show pricing for all A3i products')
        self.assertEqual(len(results), len(expected))
        urls = {call.args[0] for call in get.call_args_list}
        for page_id in expected:
            url = PRODUCT_URLS.get(page_id) or 'https://a3i-accel.com/' + ('' if page_id == 'home' else f'partials/{page_id}.html')
            self.assertIn(url, urls)
        self.assertNotIn('https://a3i-accel.com/partials/about.html', urls)

    def test_overview_fallback_keeps_all_product_evidence(self):
        pages = [{'url': f'https://a3i-accel.com/partials/product-{i}.html',
                  'content': f'Product {i}: subscription details unavailable.'} for i in range(6)]
        found = {'url': 'https://a3i-accel.com/partials/extra.html', 'content': 'Call for Pricing'}
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                patch('tools.website_results', return_value=pages), \
                patch('tools._search_tavily', return_value={'results': [found]}) as search:
            result = search_a3i_web.invoke({'query': 'all products pricing'})
        search.assert_called_once()
        self.assertEqual(len(result['results']), 7)

    def test_fallback_does_not_push_out_direct_product_offer(self):
        pages = [
            {'url': 'https://send3r.a3i-accel.com/', 'content': 'Send3r outreach.'},
            {'url': 'https://send3r.a3i-accel.com/demo',
             'content': 'Founding offer: Pro at $24 a month for 6 months, first 100 organizations.'},
        ]
        found = [{'url': f'https://a3i-accel.com/page-{i}', 'content': 'Send3r information.'}
                 for i in range(3)]
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                patch('tools.website_results', return_value=pages), \
                patch('tools._search_tavily', return_value={'results': found}):
            result = search_a3i_web.invoke({'query': 'Send3r pricing'})
        self.assertEqual(len(result['results']), 5)
        self.assertTrue(any('$24 a month for 6 months' in page['content'] for page in result['results']))

    def test_irrelevant_download_triggers_official_search(self):
        page = {'url': 'https://a3i-accel.com/', 'content': 'Welcome to our website.'}
        found = {'url': 'https://a3i-accel.com/partials/about.html',
                 'content': 'Alex is the CEO.'}
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                patch('tools.website_results', return_value=[page]), \
                patch('tools._search_tavily', return_value={'results': [found]}) as search, \
                patch('tools.discover', return_value=[]):
            result = search_a3i_web.invoke({'query': 'A3i CEO'})
        search.assert_called_once()
        self.assertEqual(result['results'][0], found)

    def test_founder_text_does_not_satisfy_ceo_or_coo_question(self):
        page = {'url': 'https://a3i-accel.com/partials/about.html',
                'content': 'Alex founded the company. Sam is a co-founder.'}
        for role in ('CEO', 'COO'):
            with self.subTest(role=role), \
                    patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                    patch('tools.website_results', return_value=[page]), \
                    patch('tools._search_tavily', return_value={'results': []}) as search, \
                    patch('tools.discover', return_value=[]):
                search_a3i_web.invoke({'query': f'A3i {role}'})
                search.assert_called_once()

    def test_matching_evidence_avoids_fallback(self):
        page = {'url': 'https://a3i-accel.com/', 'content': 'NeXtory records meetings.'}
        with patch('tools.website_results', return_value=[page]), \
                patch('tools._search_tavily') as search:
            search_a3i_web.invoke({'query': 'What is NeXtory?'})
        search.assert_not_called()

    def test_fallback_outage_preserves_downloaded_evidence(self):
        page = {'url': 'https://a3i-accel.com/', 'content': 'Our products help teams.'}
        for fallback in ({'error': 'unavailable'}, {'results': []}, RuntimeError('offline')):
            search_cache.clear()
            with self.subTest(fallback=fallback), \
                    patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                    patch('tools.website_results', return_value=[page]), \
                    patch('tools._search_tavily', **(
                        {'side_effect': fallback} if isinstance(fallback, Exception)
                        else {'return_value': fallback})):
                result = search_a3i_web.invoke({'query': 'NeXtory pricing'})
            self.assertIn('Our products help teams.', result['results'][0]['content'])

    def test_fallback_replaces_same_url_excerpt_and_is_cached(self):
        page = {'url': 'https://a3i-accel.com/', 'content': 'Welcome.'}
        found = {**page, 'content': 'NeXtory pricing: Call for Pricing.'}
        with patch.dict('os.environ', {'TAVILY_API_KEY': 'test-not-real'}), \
                patch('tools.website_results', return_value=[page]) as reader, \
                patch('tools._search_tavily', return_value={'results': [found]}) as search:
            first = search_a3i_web.invoke({'query': 'NeXtory pricing'})
            self.assertEqual(first['results'], [found])
            self.assertEqual(first, search_a3i_web.invoke({'query': 'NeXtory pricing'}))
        reader.assert_called_once()
        search.assert_called_once()

    def test_ceo_reads_about_fragment_and_keeps_founder_text(self):
        html = '<style>hidden</style><h2>Our Team</h2><h3>Alok Kumar</h3><p>Co-Founder &amp; CEO</p>'
        response = httpx.Response(200, text=html, request=httpx.Request('GET', 'https://a3i-accel.com/partials/about.html'))
        with patch('website.httpx.Client.get', return_value=response) as get:
            results = website_results('who is the CEO of A3i?')
        self.assertIn('partials/about.html', get.call_args_list[0].args[0])
        excerpt = relevant_excerpt(results[0]['content'], 'CEO of A3i')
        self.assertIn('Alok Kumar', excerpt)
        self.assertIn('CEO', excerpt)
        self.assertNotIn('hidden', excerpt)

    def test_public_content_works_without_tavily_and_is_cached(self):
        page = {'url': 'https://a3i-accel.com/partials/services.html', 'content': '# Subscriptions\nCall for Pricing'}
        with patch('tools.website_results', return_value=[page]) as reader, patch('tools.TavilySearch') as tavily:
            with patch.dict('os.environ', {'TAVILY_API_KEY': ''}):
                self.assertIn('Call for Pricing', search_a3i_web.invoke({'query': 'pricing'})['results'][0]['content'])
                search_a3i_web.invoke({'query': 'pricing'})
        reader.assert_called_once()
        tavily.assert_not_called()

    def test_fragments_fail_safely_without_using_redirects(self):
        with patch('website.httpx.Client.get', side_effect=httpx.ConnectError('offline')):
            self.assertEqual(website_results('CEO of A3i'), [])

    def test_industry_specific_question_selects_industry_content(self):
        response = httpx.Response(200, text='<h2>AI for Banks</h2><p>Industry solutions.</p>', request=httpx.Request('GET', 'https://a3i-accel.com/partials/ind-banks.html'))
        with patch('website.httpx.Client.get', return_value=response) as get:
            website_results('Does A3i-Accel provide industry-specific AI solutions?')
        self.assertTrue(any('/partials/ind-' in call.args[0] for call in get.call_args_list))

    def test_invalid_catalog_id_cannot_become_an_external_url(self):
        catalog = [{'id': 'https://example.org', 'keywords': 'CEO', 'title': 'Invalid'}]
        with patch('pathlib.Path.read_text', return_value=json.dumps(catalog)), patch('website.httpx.Client.get') as get:
            self.assertEqual(website_results('CEO'), [])
        get.assert_not_called()

    def test_parser_excludes_scripts_and_forms(self):
        parser = PageText()
        parser.feed('<h2>Contact</h2><script>secret()</script><form>private fields</form><p>Public address</p>')
        self.assertEqual(parser.text(), '## Contact\nPublic address')

    def test_product_navigation_labels_survive_extraction(self):
        parser = PageText()
        parser.feed('<h2>NeXtory</h2><p>Meeting capture.</p>'
                    '<a href="/#sol-notes-ai" data-nav="sol-notes-ai" aria-label="NeXtory: go to Notes AI"></a>'
                    '<form><a aria-label="private form value"></a></form>')
        excerpt = relevant_excerpt(parser.text(), 'NeXtory pricing')
        self.assertIn('NeXtory: go to Notes AI', excerpt)
        self.assertIn('linked product page: https://a3i-accel.com/partials/sol-notes-ai.html', excerpt)
        self.assertNotIn('private form value', excerpt)

    def test_send3r_pricing_reads_its_own_official_pages(self):
        response = httpx.Response(200, text='<h2>Founding offer</h2><p>Pro at $24 a month for 6 months.</p>',
                                  request=httpx.Request('GET', 'https://send3r.a3i-accel.com/demo'))
        with patch('website.httpx.Client.get', return_value=response) as get:
            pages = website_results('What is the price of Send3r?')
        self.assertEqual({call.args[0] for call in get.call_args_list}, set(PRODUCT_URLS.values()))
        self.assertIn('$24 a month for 6 months', pages[0]['content'])

    def test_invenio_is_not_routed_to_unrelated_ai_lead_plans(self):
        response = httpx.Response(200, text='<h2>Contact</h2>',
                                  request=httpx.Request('GET', 'https://a3i-accel.com/'))
        with patch('website.httpx.Client.get', return_value=response) as get:
            website_results('Invenio pricing')
        urls = {call.args[0] for call in get.call_args_list}
        self.assertIn('https://a3i-accel.com/partials/contact.html', urls)
        self.assertNotIn('https://a3i-accel.com/partials/prod-lead.html', urls)

    def test_multiline_headings_keep_definitions_retrievable(self):
        parser = PageText()
        parser.feed('<h2>Examples</h2><p>Apps Ants Agents used in business.</p>'
                    '<h3>\n Apps\n</h3><p>Single-purpose tools.</p>'
                    '<h3>\n Ants\n</h3><p>Reusable endpoints.</p>'
                    '<h3>\n Agents\n</h3><p>Multi-step workers.</p>')
        text = parser.text()
        self.assertIn('### Ants\nReusable endpoints.', text)
        excerpt = relevant_excerpt(text, 'What are Apps Ants and Agents?')
        self.assertIn('Reusable endpoints.', excerpt)
        self.assertIn('Multi-step workers.', excerpt)

    def test_product_overview_can_include_more_than_three_products(self):
        products = ['NeXtory', 'H3iVE', 'Send3r', 'SpeedRec', 'Invenio', 'Video AI']
        text = '\n'.join(f'## {name}\nPublished description.' for name in products)
        excerpt = relevant_excerpt(text, 'What products does A3i offer?')
        self.assertTrue(all(name in excerpt for name in products))
        self.assertLessEqual(len(excerpt), 3600)

    def test_definition_headings_beat_incidental_mentions_of_work(self):
        text = ('# Company that gets work done\nStrong execution.\n'
                '## Mundane Work\nApps Ants Agents at work.\n'
                '### Leads found while you work\nAgents do work.\n'
                '### Ants\nReusable workflow endpoints.\n'
                '### Agents\nMulti-step workers.')
        excerpt = relevant_excerpt(text, 'How do Ants work together with AI Agents?')
        self.assertIn('Reusable workflow endpoints.', excerpt)
        self.assertIn('Multi-step workers.', excerpt)

    def test_quote_keeps_real_plan_instead_of_illustrative_demo_amount(self):
        text = ('## Demo\nExample order $41.85.\n## Intro\nWelcome.\n'
                '## Features\nUseful tools.\n## Licensed on this page\nPro $29 / month.')
        self.assertIn('$29 / month', relevant_excerpt(text, 'Can I get a quote?'))

    def test_pricing_section_keeps_later_tiers_after_long_intro(self):
        text = ('## Introduction\n' + 'Welcome. ' * 200
                + '\n## Licensed on this page\nFree\n$0 / month\n'
                + 'Included meeting features. ' * 55
                + '\nBusiness\n$49 / month\n60 meeting hours\n'
                + 'Enterprise\nCustom\nCustom retention policy\nDedicated support')
        excerpt = relevant_excerpt(text, 'AI-Note pricing plans')
        self.assertIn('Business\n$49 / month\n60 meeting hours\nEnterprise', excerpt)
        self.assertIn('Custom retention policy\nDedicated support', excerpt)
        self.assertLessEqual(len(excerpt), 3600)

    def test_named_homepage_products_are_not_lost_to_different_fragment_names(self):
        response = httpx.Response(200, text='<h2>NeXtory</h2><p>Meeting captured.</p>',
                                  request=httpx.Request('GET', 'https://a3i-accel.com/'))
        for product in ('NeXtory', 'Invenio'):
            with self.subTest(product=product), patch('website.httpx.Client.get', return_value=response) as get:
                website_results(f'What does {product} do?')
                self.assertIn('https://a3i-accel.com/', [call.args[0] for call in get.call_args_list])

    def test_question_dataset_contains_all_one_hundred_questions(self):
        rows = json.loads((Path(__file__).resolve().parent.parent / 'questions.json').read_text(encoding='utf-8'))
        self.assertEqual([row['id'] for row in rows], list(range(1, 101)))
        self.assertIn('CEO', rows[90]['question'])
