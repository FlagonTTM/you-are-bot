"""Static contract for the standalone slide deck (UI checked with OpenCLI)."""
import hashlib
import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / 'YouAreBot_CU_Presentation.html'


class DeckParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.slides = []
        self.remote_dependencies = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.append(a['id'])
        if tag == 'section' and 'slide' in (a.get('class') or '').split():
            self.slides.append(a)
        if tag in ('script', 'img', 'iframe', 'link'):
            if (a.get('src') or a.get('href') or '').startswith(('http:', 'https:', '//')):
                self.remote_dependencies.append(a)


class PresentationContract(unittest.TestCase):
    def setUp(self):
        self.html = DECK.read_text()
        self.doc = DeckParser()
        self.doc.feed(self.html)

    def test_21_unique_sourced_slides(self):
        self.assertEqual(len(self.doc.slides), 21)
        self.assertEqual(len(self.doc.ids), len(set(self.doc.ids)))
        for slide in self.doc.slides:
            self.assertTrue(slide.get('data-title'))
            self.assertTrue(slide.get('data-source'))
        self.assertEqual([s['id'] for s in self.doc.slides], [f's{i}' for i in range(1, 22)])

    def test_standalone_and_no_reference_subject_leaks(self):
        self.assertFalse(self.doc.remote_dependencies)
        self.assertNotIn('https://chatgpt.com/library', self.html)
        self.assertNotIn('Tech Arena', self.html)
        self.assertNotIn('scheduler.cpp', self.html)
        self.assertIn('data:font/woff2;base64,', self.html)
        self.assertIn('prefers-reduced-motion', self.html)
        self.assertIn('@media print', self.html)
        self.assertIn('connect-src \'none\'', self.html)

    def test_metrics_are_bound_to_local_evidence(self):
        facts = json.loads((ROOT / 'evidence/session-facts.json').read_text())
        proof = json.loads((ROOT / 'evidence/ensemble-proof.json').read_text())
        block = re.search(r'<script type="application/json" id="presentation-evidence">(.*?)</script>', self.html, re.S)
        assert block is not None, 'Embedded provenance is required'
        embedded = json.loads(block.group(1))
        self.assertEqual(embedded['facts'], facts)
        self.assertEqual(embedded['ensemble'], proof)
        self.assertEqual(embedded['report_sha256'], hashlib.sha256((ROOT / 'report.html').read_bytes()).hexdigest())
        self.assertIn(str(facts['official']['public_logloss']), self.html)
        self.assertIn('Не live-рейтинг', self.html)
        self.assertIn('Первое место не достигнуто', self.html)
        self.assertIn('строгая analyst-blindness не заявляется', self.html)

    def test_model_description_omits_transport_not_method(self):
        slides = '\n'.join(re.findall(r'<section\b.*?</section>', self.html, re.S))
        self.assertNotIn('codex', slides.lower())
        self.assertIn('Использовали ансамбль CatBoost и предобученной языковой модели.', slides)
        self.assertIn('Итоговая вероятность — среднее прогнозов двух компонентов с весами 50/50.', slides)
        self.assertNotRegex(slides.lower(), r'локальн\w* (?:llm|языков\w* модел)')

    def test_controls_and_rubric_content(self):
        for control in ['prev', 'next', 'overview', 'motion', 'reading', 'fullscreen', 'help', 'sources']:
            self.assertIn(control, self.doc.ids)
        for text in ['run_ensemble.sh', 'run_ensemble.py', 'requirements.lock', 'SQLite', 'cached', 'fresh', 'Python', 'EDA']:
            self.assertIn(text, self.html)
        self.assertIn('ArrowRight', self.html)
        self.assertIn('touchend', self.html)
        self.assertIn('localStorage', self.html)
        self.assertIn('aria-modal="true"', self.html)
        self.assertIn('beforeprint', self.html)
        self.assertIn('inert', self.html)


if __name__ == '__main__':
    unittest.main()
