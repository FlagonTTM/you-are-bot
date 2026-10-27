"""Checks for the exact saved-prediction ensemble, no ML dependencies."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parent
EXPECTED = '4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639'


class EnsembleReplayTest(unittest.TestCase):
    def test_cli_reconstructs_exact_official_submission(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            output = Path(tmp) / 'submission.csv'
            result = subprocess.run(
                [sys.executable, '-I', str(PACKAGE / 'run.py'), '--output', str(output)],
                cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), EXPECTED)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            output = Path(tmp) / 'submission.csv'
            output.write_bytes(b'keep existing file')
            result = subprocess.run(
                [sys.executable, '-I', str(PACKAGE / 'run.py'), '--output', str(output)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), b'keep existing file')

    def test_corrupted_components_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            root = Path(tmp)
            for name in ['run.py', 'components.csv', 'manifest.json']:
                shutil.copyfile(PACKAGE / name, root / name)
            with (root / 'components.csv').open('ab') as file:
                file.write(b'\n')
            output = root / 'submission.csv'
            result = subprocess.run(
                [sys.executable, '-I', str(root / 'run.py'), '--output', str(output)],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SHA-256 mismatch', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
