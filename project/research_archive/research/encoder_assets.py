"""Public, pinned, credential-free model assets; offline training verification."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
MAX_BYTES = 500000000
PUBLISHER_SHA256 = {
    'model.safetensors': '1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477',
    'tokenizer.json': '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39',
}
ALLOWED = {'model.safetensors', 'config.json', 'tokenizer.json',
           'tokenizer_config.json', 'special_tokens_map.json'}


def validate_plan(plan):
    if plan.get('model_id') != MODEL_ID or plan.get('revision') != REVISION:
        raise ValueError('unapproved model/revision')
    if plan.get('license') != 'mit' or plan.get('trust_remote_code') is not False:
        raise ValueError('unapproved license/remote code')
    files = plan.get('allowlisted_files', [])
    if len(files) != len(ALLOWED) or {f.get('rfilename') for f in files} != ALLOWED:
        raise ValueError('asset allowlist mismatch')
    if any(type(f.get('size')) is not int or f['size'] <= 0 for f in files):
        raise ValueError('invalid asset byte count')
    total = sum(f['size'] for f in files)
    if (plan.get('maximum_download_bytes') != MAX_BYTES or total > MAX_BYTES or
            plan.get('expected_total_bytes') != total):
        raise ValueError('asset byte budget mismatch')
    for f in files:
        lfs = f.get('lfs')
        if lfs and (lfs.get('size') != f['size'] or
                    not re.fullmatch('[a-f0-9]{64}', str(lfs.get('sha256')))):
            raise ValueError('invalid official LFS integrity binding')
    for name in ('model.safetensors', 'tokenizer.json'):
        lfs = next(f for f in files if f['rfilename'] == name).get('lfs')
        if not lfs:
            raise ValueError('missing official LFS integrity binding')
        if lfs['sha256'] != PUBLISHER_SHA256[name]:
            raise ValueError('publisher asset digest differs from authorized revision')
    return plan


def stream_asset(source, destination, expected_size, expected_sha256=None):
    if type(expected_size) is not int or not 0 < expected_size <= MAX_BYTES:
        raise ValueError('invalid asset size budget')
    digest = hashlib.sha256()
    count = 0
    with Path(destination).open('xb') as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > expected_size:
                raise ValueError('asset size exceeds declared size')
            target.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if count != expected_size:
        raise ValueError('asset size mismatch')
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('asset SHA256 mismatch')
    return {'size': count, 'sha256': actual}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(model_dir, receipt):
    model_dir = Path(model_dir)
    if model_dir.is_symlink() or str(model_dir.resolve()) != receipt.get('model_dir'):
        raise ValueError('asset directory binding mismatch')
    plan = validate_plan(receipt['plan'])
    if receipt.get('status') != 'VERIFIED':
        raise ValueError('assets not verified')
    if (not model_dir.is_dir() or {p.name for p in model_dir.iterdir()} != ALLOWED or
            set(receipt.get('files', {})) != ALLOWED):
        raise ValueError('asset file set mismatch')
    total = 0
    for expected in plan['allowlisted_files']:
        name = expected['rfilename']
        path = model_dir / name
        actual = receipt['files'][name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError('asset file size/type mismatch: ' + name)
        if actual.get('size') != expected['size'] or file_sha(path) != actual.get('sha256'):
            raise ValueError('asset file SHA256 mismatch: ' + name)
        if name in PUBLISHER_SHA256 and actual['sha256'] != PUBLISHER_SHA256[name]:
            raise ValueError('asset publisher SHA256 mismatch: ' + name)
        total += actual['size']
    if total != plan['expected_total_bytes'] or total > MAX_BYTES:
        raise ValueError('asset total byte budget mismatch')
    return receipt


def verify_assets(model_dir: Path, receipt_path: Path) -> dict:
    """Verify every local byte; never accesses a model hub or network."""
    receipt = json.loads(Path(receipt_path).read_text())
    return _verify_manifest(model_dir, receipt)


def download(plan, model_dir, receipt_path):
    """One explicit allowlist; no auth, package install or automatic retry."""
    validate_plan(plan)
    model_dir, receipt_path = Path(model_dir), Path(receipt_path)
    if model_dir.exists() or receipt_path.exists():
        raise FileExistsError('refuse asset overwrite or automatic resend')

    class PublicHFRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, headers, newurl):
            parsed = urllib.parse.urlsplit(newurl)
            host = parsed.hostname or ''
            if (parsed.scheme != 'https' or parsed.username or parsed.password or
                    not (host in ('huggingface.co', 'hf.co') or
                         host.endswith('.huggingface.co') or host.endswith('.hf.co'))):
                raise ValueError('redirect outside public model CDN allowlist')
            return super().redirect_request(request, response, code, message, headers, newurl)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicHFRedirect())
    model_dir.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    files = {}
    # Read small config/tokenizer metadata before the one allowed weight file.
    for entry in sorted(plan['allowlisted_files'], key=lambda f: f['rfilename'] == 'model.safetensors'):
        name = entry['rfilename']
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        partial = model_dir / (name + '.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'botbench-assets/1'})
        print('DOWNLOADING', name, 'expected_bytes', entry['size'], flush=True)
        with opener.open(request, timeout=60) as source:
            actual = stream_asset(source, partial, entry['size'], entry.get('lfs', {}).get('sha256'))
        os.replace(partial, model_dir / name)
        files[name] = actual
        print('VERIFIED_FILE', name, actual['size'], actual['sha256'], flush=True)
    receipt = {'status': 'VERIFIED', 'model_id': MODEL_ID, 'revision': REVISION,
               'model_dir': str(model_dir.resolve()), 'plan': plan, 'files': files,
               'total_bytes': sum(f['size'] for f in files.values()),
               'wall_seconds': time.monotonic() - started,
               'asset_loader_sha256': file_sha(Path(__file__)),
               'credentials_used': False, 'automatic_retries': 0}
    _verify_manifest(model_dir, receipt)
    partial_receipt = receipt_path.with_suffix('.json.part')
    with partial_receipt.open('x') as target:
        json.dump(receipt, target, indent=2)
    os.replace(partial_receipt, receipt_path)
    return verify_assets(model_dir, receipt_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['download', 'verify'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'download':
        if args.plan is None:
            parser.error('--plan is required for download')
        result = download(json.loads(args.plan.read_text()), args.model_dir, args.receipt)
    else:
        result = verify_assets(args.model_dir, args.receipt)
    print(json.dumps({'status': result['status'], 'total_bytes': result['total_bytes'],
                      'revision': result['revision']}, indent=2))
