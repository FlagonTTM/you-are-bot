"""Provider-neutral documentation must retain the real external LLM boundary."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
PROVIDER_NAMES = re.compile(r'codex|gpt-6\.1-sol|openai|chatgpt', re.I)


class NeutralDocumentationContract(unittest.TestCase):
    def test_authored_markdown_omits_provider_names(self):
        documents = sorted(p for p in REPOSITORY.rglob('*.md')
                           if not any(part in ('.git', 'outputs', 'evidence') for part in p.parts))
        self.assertTrue(documents)
        for document in documents:
            with self.subTest(document=str(document.relative_to(REPOSITORY))):
                self.assertNotRegex(document.read_text(), PROVIDER_NAMES)

    def test_report_and_slides_omit_provider_names(self):
        for name in ('report.html', 'YouAreBot_CU_Presentation.html'):
            with self.subTest(document=name):
                document = (ROOT / name).read_text()
                visible = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', document,
                                 flags=re.S | re.I)
                self.assertNotRegex(visible, PROVIDER_NAMES)
                self.assertIn('LLM', visible)

    def test_fresh_still_declares_external_processing_and_real_configuration(self):
        document = (ROOT / 'ENSEMBLE.md').read_text()
        self.assertIn('внешнему провайдеру', document)
        self.assertIn('llm/config.json', document)
        self.assertIn('llm/adapter.py', document)
        self.assertIn('не гарантирует', (REPOSITORY / 'README.md').read_text())


if __name__ == '__main__':
    unittest.main()
