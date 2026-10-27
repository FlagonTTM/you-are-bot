"""Coursework rubric coverage checks for the actual HTML generator."""
import importlib.util
from pathlib import Path
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('coursework_html', PROJECT / 'tools/build_html.py')
if spec is None or spec.loader is None:
    raise RuntimeError('HTML builder import specification is unavailable')
html_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(html_module)


class RubricTest(unittest.TestCase):
    def test_report_covers_training_and_both_ensemble_modes(self):
        document, payload = html_module.build_document(PROJECT)
        self.assertIn('Обработка данных и обучение · Python', document)
        self.assertIn('./project/run_ensemble.sh', document)
        self.assertIn('--llm-mode cached', document)
        self.assertIn('--llm-mode fresh', document)
        self.assertIn('run_ensemble.py', document)
        self.assertIn('сохранённые ответы LLM', document)
        self.assertIn('requirements.lock', document)
        self.assertIn('ensemble_proof', payload)
        html_module.validate_document(document)


if __name__ == '__main__':
    unittest.main()
