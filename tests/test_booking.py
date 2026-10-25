import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import leads
from chat_cli import main, request_meeting
from leads import DemoRequest, meeting_booking_url, save_demo
from router import classify_intent


class BookingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = patch.object(leads, 'DB_PATH', Path(directory.name) / 'leads.sqlite3')
        database.start()
        self.addCleanup(database.stop)
        environment = patch.dict('os.environ', {'MEETING_BOOKING_URL': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()

    def answers(self, confirmation='yes', email='visitor@example.com', time=None):
        return ['Visitor', 'Example Company', email, 'Discuss meeting notes',
                self.future if time is None else time, confirmation]

    async def test_confirmed_request_is_saved_with_contact_and_time(self):
        with patch('builtins.input', side_effect=self.answers()), patch('builtins.print'):
            result = await request_meeting()
        self.assertIn('Meeting request saved', result)
        self.assertIn('not confirmed', result)
        with sqlite3.connect(leads.DB_PATH) as db:
            row = db.execute('SELECT name, email, preferred_time FROM leads').fetchone()
        self.assertEqual(row, ('Visitor', 'visitor@example.com', self.future))

    async def test_declining_or_cancelling_never_creates_database(self):
        for answers in (['cancel'], self.answers(confirmation='no')):
            with patch('builtins.input', side_effect=answers), patch('builtins.print'):
                self.assertIn('cancelled', await request_meeting())
            self.assertFalse(leads.DB_PATH.exists())

    async def test_invalid_contact_time_and_past_time_are_rejected(self):
        for answers in (
            self.answers(email='invalid'), self.answers(time='2099-01-01T12:00'),
            self.answers(time='2000-01-01T12:00+05:30'),
        ):
            with patch('builtins.input', side_effect=answers), patch('builtins.print'):
                self.assertIn('not saved', await request_meeting())
            self.assertFalse(leads.DB_PATH.exists())

    async def test_database_failure_is_reported_without_success(self):
        with patch('builtins.input', side_effect=self.answers()), patch('builtins.print'), \
                patch('chat_cli.save_demo', side_effect=sqlite3.OperationalError('private details')):
            result = await request_meeting()
        self.assertIn('Could not save', result)
        self.assertNotIn('private details', result)

    async def test_booking_link_skips_local_collection(self):
        with patch.dict('os.environ', {'MEETING_BOOKING_URL': 'https://example.com/book'}), \
                patch('builtins.input') as prompt:
            result = await request_meeting()
        self.assertIn('https://example.com/book', result)
        prompt.assert_not_called()
        self.assertFalse(leads.DB_PATH.exists())

    async def test_terminal_booking_typo_bypasses_model_and_history(self):
        inputs = ['schedule a meating for me', *self.answers(), 'bye']
        with patch('builtins.input', side_effect=inputs), patch('builtins.print'), \
                patch('chat_cli.stream_chat') as model, patch('chat_cli.save_history') as history:
            await main()
        model.assert_not_called()
        history.assert_not_called()
        self.assertTrue(leads.DB_PATH.exists())

    def test_routing_accepts_meeting_typos_but_keeps_pricing_questions(self):
        self.assertEqual(classify_intent('schedule a meating for me'), 'demo')
        self.assertEqual(classify_intent('book a meting'), 'demo')
        self.assertEqual(classify_intent('What does a demo cost?'), 'pricing')

    def test_booking_url_rejects_unsafe_or_invalid_values(self):
        for value in ('javascript:alert(1)', 'http://example.com', 'https://user:pass@example.com',
                      'https://example.com:bad/book', 'https://example.com/book now'):
            with patch.dict('os.environ', {'MEETING_BOOKING_URL': value}):
                self.assertEqual(meeting_booking_url(), '')

    def test_existing_database_is_preserved_and_requests_are_idempotent(self):
        with sqlite3.connect(leads.DB_PATH) as db:
            db.execute('CREATE TABLE leads (id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, '
                       'name TEXT NOT NULL, company TEXT NOT NULL, need TEXT NOT NULL)')
            db.execute('INSERT INTO leads VALUES (?, ?, ?, ?, ?)',
                       ('old', str(uuid4()), 'Existing Visitor', 'Existing Company', 'Existing need'))
        request = DemoRequest(name='New Visitor', company='Company', need='Demo',
                              email='new@example.com', preferred_time=self.future,
                              request_id=uuid4(), confirmed=True)
        first = save_demo(request)
        self.assertEqual(save_demo(request), first)
        with self.assertRaises(ValueError):
            save_demo(request.model_copy(update={'email': 'changed@example.com'}))
        with sqlite3.connect(leads.DB_PATH) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM leads').fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT name FROM leads WHERE id='old'").fetchone()[0], 'Existing Visitor')
