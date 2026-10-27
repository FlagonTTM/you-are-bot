#!/usr/bin/env python3
"""Raw official inputs -> native six fits -> explicit cached/fresh LLM -> fixed blend."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from llm import adapter
import botbench
import experiment
from dependencies import check_environment
from guards import (MEMBERSHIP_SHA, assert_unchanged, cv_splits, data_manifest, sha256,
                    source_manifest, strict_json, validate_counts, validate_data, validate_membership)
from pipeline import run as native_run
from tracking import RunTracker, no_symlink

MODELS = ['fold-0', 'fold-1', 'fold-2', 'fold-3', 'fold-4', 'full']
COMPONENTS_SHA = 'ed36be23cfb492ac2dfdd77298bf998b64c900703db3b06f97af2a4e4bb6ab6e'
CACHE_MANIFEST_SHA = '382bc9a1381599993ee6784f287f09f8f99a8afab2a99dc70a8f7c40df6cc6e8'
SUBMISSION_SHA = '4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639'


def write_ensemble_csv(path, sample_ids, native, remote, *, expected_sha=None):
    if len(sample_ids) != len(set(sample_ids)) or set(native) != set(sample_ids) or set(remote) != set(sample_ids):
        raise ValueError('ensemble ID/order/cardinality mismatch')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\r\n')
    writer.writerow(['ID', 'is_bot'])
    for identifier in sample_ids:
        a, b = adapter.probability(native[identifier]), adapter.probability(remote[identifier], strict=True)
        writer.writerow([identifier, 0.5 * a + 0.5 * b])
    payload = stream.getvalue().encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('cached ensemble differs from historical submission SHA')
    with Path(path).open('xb') as file:
        file.write(payload)
    if Path(path).read_bytes() != payload:
        raise ValueError('final CSV byte readback mismatch')
    return dict(rows=len(sample_ids), sha256=digest, encoding='UTF-8', line_terminator='CRLF',
                weights=dict(catboost=0.5, llm=0.5), submitted=False)


def runtime_manifest():
    result = source_manifest(ROOT)
    names = ['run_ensemble.py', 'run_ensemble.sh', 'llm/adapter.py', 'llm/bridge.py',
             'llm/historical_bridge.py', 'llm/config.json', 'ensemble/components.csv', 'ensemble/manifest.json']
    for name in names:
        path = ROOT / name
        no_symlink(path)
        if not path.is_file():
            raise ValueError('missing required ensemble runtime source')
        result[name] = sha256(path)
    return result


def render_index(run, mode, report=None):
    """Static offline artifact index; terminal status remains in SQLite/JSON."""
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CatBoost + LLM</title>
<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f3ee;padding:16px}a{color:#135a6d}</style>
<h1>CatBoost + LLM — MODE</h1><p>RUN. Backend: stdlib SQLite + JSON, не MLflow. Не отправлено на соревнование.</p>
<p>Режим cached заново обучает native-модель, но использует исторические LLM-вероятности только после точной привязки priors/input/prompt. Fresh выполняет новые внешние inference-вызовы. CatBoost OOF — не ensemble CV; актуальный официальный score здесь не измеряется.</p>
<p><a href="tracking.json">tracking.json — конечный статус</a> · <a href="tracking.sqlite3">SQLite</a> · <a href="report.json">report.json (если этап завершён)</a></p>
<pre id="receipt">Откройте tracking.json напрямую при просмотре через file://. Для динамического индекса используйте локальный HTTP-viewer.</pre><h2>Артефакты текущего запуска</h2><ul id="artifacts"></ul>
<script>fetch('tracking.json').then(r=>r.json()).then(r=>{document.getElementById('receipt').textContent=JSON.stringify({status:r.status,error:r.error,params:r.params,metrics:r.metrics},null,2);for(const name of Object.keys(r.artifacts)){let a=document.createElement('a');a.textContent=name;a.href=name.split('/').map(encodeURIComponent).join('/');let li=document.createElement('li');li.append(a);document.getElementById('artifacts').append(li)}}).catch(()=>{});</script></html>'''
    page = page.replace('MODE', html.escape(mode)).replace('RUN.', html.escape(run.run_id) + '.')
    if report is not None:
        receipt = run._export()
        names = sorted(set(receipt['artifacts']) | {'tracking.json','tracking.sqlite3','index.html'})
        links = ''.join('<li><a href="' + html.escape(name,quote=True) + '">' + html.escape(name) + '</a></li>' for name in names)
        static = '<h2>Отчёт этого запуска (offline)</h2><pre>' + html.escape(json.dumps(report,ensure_ascii=False,indent=2)) + '</pre>'
        static += '<h2>Все параметры и метрики (offline)</h2><pre>' + html.escape(json.dumps(dict(params=receipt['params'],metrics=receipt['metrics']),ensure_ascii=False,indent=2)) + '</pre>'
        page = page.replace('</html>', static + '<h2>Все артефакты (offline)</h2><ul>' + links + '</ul></html>')
    with run.path('index.html').open('w' if report is not None else 'x', encoding='utf-8') as stream:
        stream.write(page)


def verify_native_receipt(directory):
    receipt = strict_json(directory / 'tracking.json')
    with sqlite3.connect((directory / 'tracking.sqlite3').as_uri() + '?mode=ro', uri=True) as sql:
        row = sql.execute('SELECT run_id,status FROM runs').fetchall()
        params = {n: json.loads(v) for n, v in sql.execute('SELECT name,value_json FROM params')}
        metrics = dict(sql.execute('SELECT name,value FROM metrics'))
        artifacts = {p:dict(sha256=h,bytes=n) for p,h,n in sql.execute('SELECT path,sha256,bytes FROM artifacts')}
    if row != [(receipt['run_id'], 'FINISHED')] or receipt['status'] != 'FINISHED':
        raise ValueError('native subrun did not finish')
    if params != receipt['params'] or metrics != receipt['metrics'] or artifacts != receipt['artifacts']:
        raise ValueError('native SQLite/JSON readback mismatch')
    for name, record in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('native artifact escapes owned namespace')
        path = directory / relative
        no_symlink(path)
        if sha256(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('native artifact readback hash mismatch')
    for name in MODELS:
        model = params.get('model/' + name, {})
        if model.get('tree_count') != 1000 or model.get('get_all_params', {}).get('random_seed') != 1729:
            raise ValueError('required six fitted 1000-tree seed-1729 model parameter groups missing')
        if 'get_params' not in model or not (directory / f'models/{name}.cbm').is_file():
            raise ValueError('native model/checkpoint missing')
    for name in ('models/full.cbm', 'native-test.csv', 'report.json'):
        if name not in artifacts:
            raise ValueError('required native artifact missing')
    return receipt


def execute_chain(run, data_dir, train, test, sample_ids, mode, config, manifest, components, check_immutable, cli=None):
    """Internal composition seam tested with explicitly synthetic native outputs.

    Public run() always applies raw official hashes/counts/membership/environment
    guards; this function never substitutes a cached native model.
    """
    if mode not in {'cached', 'fresh'}:
        raise ValueError('explicit cached/fresh mode required')
    started = time.perf_counter()
    if not run.path('index.html').exists():
        render_index(run, mode)
    run.params('execution', dict(llm_mode=mode, native='five grouped CV fits + one all-TRAIN full refit',
                               fresh_native_training=True, fresh_llm_inference=mode=='fresh',
                               cached_llm_replay=mode=='cached', submitted=False))
    run.event('native_cv_and_full_refit')
    native_run(Path(data_dir), run.path('native'), 'catboost')
    child = run.path('native/catboost')
    native_receipt = verify_native_receipt(child)
    for key, value in native_receipt['params'].items():
        run.params('catboost/' + key, value)
    for key, value in native_receipt['metrics'].items():
        prefix = 'catboost_oof/' + key.removeprefix('oof/') if key.startswith('oof/') else 'catboost_' + key
        run.metric(prefix, value)
    run.params('native_subrun', dict(namespace='native/catboost', status='FINISHED',
                    sql_json_readback_equal=True, tracking_sha256=sha256(child/'tracking.json'),
                    sqlite_sha256=sha256(child/'tracking.sqlite3'), parameter_groups=6))
    native_report = strict_json(child / 'report.json')
    if (native_report['counts']['train_rows'] != len(train) or native_report['counts']['test_rows'] != len(test)
            or native_report['checkpoint_replay']['all_exact'] is not True):
        raise ValueError('native counts/checkpoint replay mismatch')
    priors = botbench.read_probabilities(child / 'native-test.csv')
    if list(priors) != list(sample_ids):
        raise ValueError('native test ID/order/cardinality mismatch')
    check_immutable()
    run.event('prepare_blind_llm_payloads')
    batches = adapter.prepare_batches(train, test, priors, run.path('llm'))
    locks = {b.name: adapter.batch_lock(b) for b in batches}
    run.write_json('llm-payload-locks.json', locks)
    run.params('llm_preparation', dict(training_rows=len(train), evaluated_rows=len(test),
                   evaluated_labels_provided=False, batch_count=len(batches), batch_size_dialogs=40,
                   batching='sorted dialog_id chunks; original test participant row order within chunks',
                   prompt_template_sha256=hashlib.sha256(adapter.bridge.PROMPT.encode('utf-8')).hexdigest(),
                   model_requested=adapter.bridge.MODEL, configuration=config,
                   truncated_dialogues=sum(sum(bool(d['input_truncated']) for d in strict_json(b/'input.json')) for b in batches)))
    check_immutable()
    if mode == 'cached':
        run.event('bind_historical_llm_cache')
        binding = adapter.bind_cache(batches, manifest, priors, components)
        run.write_json('cache-binding.json', binding)
        remote = {i: components[i]['llm'] for i in sample_ids}
        origin = dict(mode='historical_llm_probabilities_exact_payload_bound',
                      source_batch_digests=manifest['source_batch_digests'], new_llm_calls=0,
                      historical_source_generation_bridge_sha256=manifest.get('source_generation_bridge_sha256'),
                      historical_generation_manifest_sha256=manifest.get('source_generation_manifest_sha256'))
    else:
        if cli is None:
            raise ValueError('fresh CLI preflight receipt required')
        remote = {}
        origins = {}
        for batch in batches:
            run.event('fresh_llm_' + batch.name)
            predictions = adapter.fresh_infer(batch, cli, check_immutable, lock=locks[batch.name])
            for prediction in predictions:
                for side in (0, 1):
                    identifier = prediction['dialog_id'] + '_' + str(side)
                    if identifier in remote:
                        raise ValueError('duplicate fresh LLM participant ID')
                    remote[identifier] = prediction['p' + str(side)]
            origins[batch.name] = {name: sha256(batch/name) for name in ('input.json','prompt.txt','output.json','events.jsonl','receipt.json','boundary-receipt.json')}
        origin = dict(mode='fresh_external_model_inference', new_llm_calls=len(batches), batches=origins,
                      deterministic_external_responses_guaranteed=False)
    if any(adapter.batch_lock(b) != locks[b.name] for b in batches):
        raise ValueError('LLM payload changed during inference/replay')
    check_immutable()
    run.params('llm_origin', origin)
    with run.path('llm-test.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\r\n'); writer.writerow(['ID','is_bot'])
        if set(remote) != set(sample_ids):
            raise ValueError('LLM test membership mismatch')
        writer.writerows((i, adapter.probability(remote[i], strict=True)) for i in sample_ids)
    if botbench.read_probabilities(run.path('llm-test.csv')) != remote:
        raise ValueError('LLM probability CSV readback mismatch')
    run.event('assemble_fixed_50_50')
    validation = write_ensemble_csv(run.path('submission.csv'), sample_ids, priors, remote,
                                   expected_sha=config['submission_sha256'] if mode=='cached' else None)
    if list(botbench.read_probabilities(run.path('submission.csv'))) != sample_ids:
        raise ValueError('final sample order readback mismatch')
    report = dict(schema='coursework.ensemble.v1', run_id=run.run_id, llm_mode=mode,
                  native_subrun='native/catboost', native_training_fits=6,
                  counts=dict(training_participants=len(train),test_participants=len(test),llm_batches=len(batches)),
                  catboost_oof_metrics=native_report['metrics'], ensemble_cv_evaluated=False,
                  ensemble_cv_note='No LLM OOF labels/predictions generated; CatBoost OOF is NOT ensemble CV.',
                  validation=validation, test_labels_read=False, test_payloads_blind=True,
                  runtime_seconds=time.perf_counter()-started, submitted=False,
                  historical_csv_matches=validation['sha256']==config['submission_sha256'],
                  historical_hash_is_success_gate=mode=='cached',
                  historical_public_logloss=manifest.get('historical_public_logloss') if mode=='cached' else None,
                  public_score_recomputed_locally=False, fresh_official_score=None,
                  synthetic_fixture=bool(native_report.get('synthetic_fixture',False)),
                  tracking_backend='stdlib SQLite + JSON (not MLflow)',
                  raw_payload_policy='Local ignored outputs only; do not export raw texts/events to Git or evidence bundle.')
    run.write_json('report.json', report)
    run.write_json('submission-validation.json', validation)
    run.collect_artifacts()
    render_index(run, mode, report)
    return report


def validate_output_location(data_dir, output_dir):
    resolved = Path(output_dir).resolve()
    if resolved.is_relative_to(Path(data_dir).resolve()):
        raise ValueError('output must be outside official data directory')
    if resolved.is_relative_to(ROOT.parent) and not resolved.is_relative_to(ROOT/'outputs'):
        raise ValueError('raw payload output inside coursework must use ignored project/outputs')


def run(data_dir, output_dir, run_id, mode):
    if mode not in {'cached','fresh'}:
        raise ValueError('explicit --llm-mode cached|fresh required')
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).absolute()
    validate_output_location(data_dir,output_dir)
    with RunTracker(output_dir, run_id) as tracking:
        render_index(tracking, mode)
        tracking.params('requested_mode', mode)
        tracking.event('preflight_source_and_scientific_parity')
        before = runtime_manifest()
        for module, path in [(botbench,'src/botbench.py'), (experiment,'src/experiment.py'), (adapter,'llm/adapter.py'), (adapter.bridge,'llm/bridge.py')]:
            if Path(module.__file__).resolve() != ROOT/path:
                raise ValueError('outside-project application import refused')
        config = strict_json(ROOT/'llm/config.json')
        if (config['model']!='gpt-6.1-sol' or config['reasoning_effort']!='high' or config['codex_cli_version']!='0.163.0'
                or config['weights']!={'catboost':0.5,'llm':0.5} or config['dialogs_per_batch']!=40
                or config['maximum_batches']!=10 or config['timeout_seconds']!=601
                or config['components_sha256']!=COMPONENTS_SHA or config['cache_manifest_sha256']!=CACHE_MANIFEST_SHA
                or config['submission_sha256']!=SUBMISSION_SHA):
            raise ValueError('frozen ensemble configuration mismatch')
        parity = adapter.verify_sources()
        if parity['historical_bridge_sha256'] != 'dc95650cc3cf3081e87c2849a03d3f332c6b43e0fbdca5e95167d57253b118d6':
            raise ValueError('historical generation source identity differs')
        if before['ensemble/manifest.json']!=CACHE_MANIFEST_SHA:
            raise ValueError('historical cache manifest hash mismatch')
        manifest = strict_json(ROOT/'ensemble/manifest.json')
        if (manifest['components_sha256']!=COMPONENTS_SHA or manifest['submission_sha256']!=SUBMISSION_SHA
                or manifest['verified_llm_batches']!=10 or manifest['rows']!=758
                or manifest['llm_model_requested']!=config['model'] or manifest['weights']!=config['weights']):
            raise ValueError('historical cache recipe mismatch')
        tracking.params('source_scientific_parity',parity)
        tracking.params('ensemble_configuration',config)
        tracking.params('cache_manifest',manifest)
        tracking.params('runtime_source_config_sha256',before)
        for name, digest in before.items():
            target=tracking.path('snapshot/'+name); target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if sha256(target)!=digest: raise ValueError('runtime source changed during snapshot')
        tracking.event('preflight_environment')
        tracking.params('environment',check_environment(ROOT))
        cli=None
        if mode=='fresh':
            tracking.event('preflight_codex')
            cli=adapter.codex_preflight()
            tracking.params('codex_cli',cli)
        else:
            tracking.params('codex_cli',dict(required_for_mode=False, executed=False, version_pin_for_fresh='0.163.0'))
        tracking.event('preflight_official_data')
        train,test,sample_ids,input_before=validate_data(data_dir)
        if sha256(ROOT/'configs/membership-v2.json')!=MEMBERSHIP_SHA:
            raise ValueError('frozen membership SHA mismatch')
        folds=validate_membership(train,strict_json(ROOT/'configs/membership-v2.json'))
        counts=validate_counts(train,test,folds); list(cv_splits(train,folds))
        tracking.params('data_counts',counts)
        tracking.params('input_sha256',input_before)
        tracking.write_json('inputs-manifest.json',dict(before=input_before,official_hashes_match=True))
        components=adapter.load_components(ROOT/'ensemble/components.csv',sample_ids,COMPONENTS_SHA)
        def immutable():
            assert_unchanged(input_before,data_manifest(data_dir),'official raw inputs')
            assert_unchanged(before,runtime_manifest(),'consumed runtime source/config/cache')
        immutable()
        report=execute_chain(tracking,data_dir,train,test,sample_ids,mode,config,manifest,components,immutable,cli)
        tracking.event('verify_immutable_inputs_and_source')
        immutable()
        tracking.write_json('immutability.json',dict(source_before=before,source_after=runtime_manifest(),
                        input_before=input_before,input_after=data_manifest(data_dir),all_unchanged=True,
                        closure_note='Consumed native + ensemble runtime/config/cache/lock only. Mutable HTML/docs/evidence excluded.'))
        tracking.collect_artifacts()
        render_index(tracking,mode,report)
        tracking.finish()
        print(json.dumps(dict(run_id=tracking.run_id,status=tracking.status,llm_mode=mode,
                             submission_sha256=report['validation']['sha256'],submitted=False)),flush=True)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Five byte-unchanged official inputs')
    parser.add_argument('--llm-mode',choices=['cached','fresh'],required=True,help='Explicit historical LLM replay or new external inference')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs',help='Parent of a NEW immutable run directory')
    parser.add_argument('--run-id',default=None,help='Unique run namespace; never overwritten')
    args=parser.parse_args(argv)
    try:
        run(args.data_dir,args.output_dir,args.run_id,args.llm_mode)
    except (Exception,KeyboardInterrupt) as exc:
        print(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,
              hint='Read tracking.json error.type/error.stage. No automatic retry or cached/fresh substitution.')),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
