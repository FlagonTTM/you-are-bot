"""Frozen native template baseline. No HPO, APIs, LLMs or official uploads."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

import botbench
import experiment
from dependencies import check_environment
from eda import summarize
from guards import (MEMBERSHIP_SHA, OFFICIAL_COUNTS, OFFICIAL_HASHES,
                    assert_unchanged, blind_rows, cv_splits, data_manifest, sha256,
                    source_manifest, strict_json, validate_counts, validate_data,
                    validate_membership)
from tracking import RunTracker, dumps

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_OOF = 0.16919850572003053


def scores(rows, probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray([r['y'] for r in rows])
    return {'logloss': float(log_loss(labels, values, labels=[0, 1])),
            'auc': float(roc_auc_score(labels, values))}


def save_checkpoint(model, path):
    if path.exists():
        raise FileExistsError(path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path), format='cbm')
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('empty native checkpoint')


def replay_checkpoint(path, rows, expected) -> dict:
    """Read only self-created native CatBoost checkpoints; never load pickles."""
    loaded = CatBoostClassifier()
    loaded.load_model(str(path), format='cbm')
    actual = np.asarray(experiment.predict('template', loaded, blind_rows(rows)), dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    exact = actual.shape == expected.shape and np.array_equal(actual, expected)
    if not exact:
        raise ValueError('checkpoint probability replay differs')
    return dict(exact=True, rows=len(rows), max_absolute_difference=float(np.max(np.abs(actual - expected))))


def read_submission(path, expected_ids):
    values = botbench.read_probabilities(path)
    identifiers = list(values)
    if identifiers != list(expected_ids):
        raise ValueError('submission ID/order/cardinality mismatch')
    return identifiers, np.asarray(list(values.values()), dtype=np.float64)


def model_receipt(model):
    return dict(get_params=model.get_params(), get_all_params=model.get_all_params(),
                tree_count=int(model.tree_count_), classes=model.classes_.tolist(),
                note='get_params includes explicit constructor/thread/categorical settings; get_all_params includes fitted effective defaults.')


def render_index(run, report):
    receipt = run._export()
    links = ['tracking.json', 'tracking.sqlite3', *receipt['artifacts'].keys()]
    links = sorted(set(links))
    items = ''.join(f'<li><a href="{html.escape(name, quote=True)}">{html.escape(name)}</a></li>' for name in links)
    page = f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native baseline — {html.escape(run.run_id)}</title>
<style>body{{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#faf9f6;color:#192322}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eeeee8;padding:18px}}a{{color:#135a6d}}h1,h2{{line-height:1.2}}</style>
<h1>Native CatBoost: {html.escape(run.run_id)}</h1>
<p>Реальные результаты этого запуска. Авторитетный конечный статус: <a href="tracking.json">tracking.json</a> / SQLite.
Трекинг stdlib SQLite + JSON, не MLflow. HTTP-сервис не реализован. Данные не отправлялись.</p>
<h2>Метрики и границы</h2><pre>{html.escape(dumps(report))}</pre>
<h2>Все записанные параметры и эффективные defaults</h2><pre>{html.escape(dumps(receipt['params']))}</pre>
<h2>Артефакты</h2><p>Финальные SHA-256 и размеры — в tracking.json. Сырые данные не экспортируются; OOF/CSV содержат необходимые ID.</p><ul>{items}</ul>
</html>'''
    path = run.path('index.html')
    with path.open('x', encoding='utf-8') as f:
        f.write(page)


def execute_pipeline(run, train, test, folds, sample_ids):
    """Validated data -> five development fits + one frozen full-training fit.

    Low-level API is exercised on artificial tiny fixtures by unit tests. Only
    run() is the official CLI entry and it always enforces official hashes.
    """
    started = time.perf_counter()
    if any('y' in row or 'is_bot' in row for row in test):
        raise ValueError('unexpected test target')
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != {r['ID'] for r in test}:
        raise ValueError('sample ID/cardinality mismatch')
    splits = list(cv_splits(train, folds))
    run.params('scientific_protocol', dict(model='template', explicit_model_params=experiment.BEHAVIOR_CONFIG,
               n_folds=5, early_stopping=False, eval_set_passed_to_fit=False,
               cv_scope='fold>=0 development only; entire connected component excluded',
               confirmation='fold=-1 excluded from CV and selection; included only in frozen final full refit',
               metric_dtype='float64, sklearn log_loss(labels=[0,1]), no custom clipping',
               submission='native full-refit probabilities only; NOT SUBMITTED',
               http_service_implemented=False))
    run.event('eda')
    run.write_json('eda.json', summarize(train, test, folds))
    # Assert full strong bank instead of silently substituting a smaller bank.
    probe = experiment.model_input(blind_rows(train[:2]), 'template')
    if probe.shape[1] != 176:
        raise ValueError('strong template bank must contain 162 numeric + 14 categorical columns')
    feature_counts = dict(numeric=162, categorical=14, total=int(probe.shape[1]))
    run.params('features', dict(**feature_counts, cat_features=list(range(162, 176)),
                              uuid_values='masked by clean_text for categorical features',
                              identity_columns_as_features=False, targets_as_features=False))
    probabilities = np.full(len(train), np.nan, dtype=np.float64)
    receipts, replay = [], []
    for fold, itrain, ival in splits:
        begin = time.perf_counter()
        training = [train[i] for i in itrain]
        validation = [train[i] for i in ival]
        run.event(f'fit_fold_{fold}', training_rows=len(training), validation_rows=len(validation))
        predicted, model = experiment.fit_predict('template', training, blind_rows(validation))
        probabilities[ival] = predicted
        checkpoint = f'models/fold-{fold}.cbm'
        save_checkpoint(model, run.path(checkpoint))
        run.params(f'model/fold-{fold}', model_receipt(model))
        checked = replay_checkpoint(run.path(checkpoint), validation, predicted)
        checked.update(checkpoint=checkpoint, scope=f'fold-{fold}')
        replay.append(checked)
        metrics = scores(validation, predicted)
        for key, value in metrics.items():
            run.metric(f'fold-{fold}/{key}', value)
        receipt = dict(fold=fold, train_rows=len(training), validation_rows=len(validation),
                       train_IDs=[r['ID'] for r in training], validation_IDs=[r['ID'] for r in validation],
                       train_groups=sorted({r['group'] for r in training}),
                       validation_groups=sorted({r['group'] for r in validation}),
                       component_overlap=len({r['group'] for r in training} & {r['group'] for r in validation}),
                       confirmation_rows_in_fit=int(np.sum(np.asarray(folds)[itrain] == -1)),
                       metrics=metrics, seconds=time.perf_counter() - begin, checkpoint=checkpoint)
        receipts.append(receipt)
        print(json.dumps(dict(stage='fold_finished', fold=fold, metrics=metrics)), flush=True)
        del model
    idx = np.flatnonzero(folds >= 0)
    dev = [train[i] for i in idx]
    experiment.write_predictions(run.path('oof.csv'), dev, probabilities[idx], folds[idx])
    saved_oof = botbench.read_probabilities(run.path('oof.csv'), has_fold=True)
    if list(saved_oof) != [r['ID'] for r in dev] or not np.array_equal(list(saved_oof.values()), probabilities[idx]):
        raise ValueError('saved OOF probability/ID parity failed')
    metrics = scores(dev, probabilities[idx])
    for key, value in metrics.items():
        run.metric(f'oof/{key}', value)
    run.write_json('fold-receipts.json', receipts)
    run.event('full_refit', train_rows=len(train), test_rows=len(test))
    by_id = {r['ID']: r for r in test}
    ordered_test = [by_id[identifier] for identifier in sample_ids]
    full_probabilities, full_model = experiment.fit_predict('template', train, blind_rows(ordered_test))
    save_checkpoint(full_model, run.path('models/full.cbm'))
    run.params('model/full', model_receipt(full_model))
    experiment.write_predictions(run.path('native-test.csv'), ordered_test, full_probabilities)
    _, saved_test = read_submission(run.path('native-test.csv'), sample_ids)
    if not np.array_equal(saved_test, full_probabilities):
        raise ValueError('saved native-test CSV probability parity failed')
    checked = replay_checkpoint(run.path('models/full.cbm'), ordered_test, saved_test)
    checked.update(checkpoint='models/full.cbm', scope='full-refit-test')
    replay.append(checked)
    replay_report = dict(all_exact=all(r['exact'] for r in replay),
                         comparison='numpy.array_equal, float64; no tolerance relaxation', models=replay)
    run.write_json('checkpoint-replay.json', replay_report)
    report = dict(schema='coursework.native.v1', run_id=run.run_id, model='template',
                  counts=dict(train_rows=len(train), oof_rows=len(dev), test_rows=len(ordered_test),
                              confirmation_rows=int(np.sum(folds == -1))),
                  feature_counts=feature_counts, metrics=metrics,
                  fold_metrics=[dict(fold=r['fold'], metrics=r['metrics']) for r in receipts],
                  historical_comparison=dict(expected_oof_logloss=HISTORICAL_OOF,
                      actual_minus_historical=metrics['logloss'] - HISTORICAL_OOF,
                      matches_at_1e_12=abs(metrics['logloss'] - HISTORICAL_OOF) <= 1e-12,
                      note='Historical diagnostic reference, not the measured value of this run. Not a tuning gate.'),
                  checkpoint_replay=replay_report, seconds=time.perf_counter() - started,
                  confirmation_used_for_selection=False, test_labels_read=False,
                  final_fit_uses_all_training_labels=True, submitted=False,
                  http_service_implemented=False,
                  official_score_note='Historical 0.1303426569 belongs to a separate LLM ensemble; it is NOT the score of this native CSV.')
    run.write_json('report.json', report)
    run.event('pipeline_artifacts_ready', seconds=report['seconds'])
    run.collect_artifacts()
    render_index(run, report)
    return report


def git_context(root):
    """Read HEAD/dirty summary only. No remotes, credentials, fetches or Git writes."""
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        if head.returncode or not re.fullmatch(r'[0-9a-f]{40,64}', head.stdout.strip()):
            return {'available': False}
        status = subprocess.run(['git', '--no-optional-locks', '-C', str(root), 'status', '--porcelain', '--untracked-files=normal'],
                                capture_output=True, text=True, timeout=10)
        if status.returncode:
            return {'available': True, 'head': head.stdout.strip(), 'dirty': 'unknown'}
        return {'available': True, 'head': head.stdout.strip(), 'dirty': bool(status.stdout),
                'changed_entry_count': len(status.stdout.splitlines())}
    except (OSError, subprocess.TimeoutExpired):
        return {'available': False}


def run(data_dir: Path, output_dir: Path, run_id=None):
    with RunTracker(output_dir, run_id) as tracking:
        tracking.event('preflight_source')
        before = source_manifest(ROOT)
        config = strict_json(ROOT / 'configs/template.json')
        origins = strict_json(ROOT / 'src/origins.json')
        for name, record in origins['files'].items():
            if before.get(f'src/{name}') != record['sha256']:
                raise ValueError('frozen original source-copy SHA mismatch')
        if Path(botbench.__file__).resolve() != ROOT / 'src/botbench.py' or Path(experiment.__file__).resolve() != ROOT / 'src/experiment.py':
            raise ValueError('outside-project code import refused')
        if (config['model'] != 'template' or config['model_params'] != experiment.BEHAVIOR_CONFIG
                or config['feature_counts'] != dict(numeric=162, categorical=14)
                or config['official_hashes'] != OFFICIAL_HASHES or config['counts'] != OFFICIAL_COUNTS
                or config['membership_sha256'] != MEMBERSHIP_SHA or config['folds'] != 5
                or config['historical_oof_logloss'] != HISTORICAL_OOF):
            raise ValueError('frozen configuration contract mismatch')
        tracking.params('configuration', config)
        tracking.params('source_origins', origins)
        tracking.params('git_context', git_context(ROOT))
        tracking.event('preflight_environment')
        tracking.params('environment', check_environment(ROOT))
        tracking.event('preflight_data')
        train, test, sample_ids, input_before = validate_data(Path(data_dir))
        if sha256(ROOT / 'configs/membership-v2.json') != MEMBERSHIP_SHA:
            raise ValueError('membership SHA mismatch')
        membership = strict_json(ROOT / 'configs/membership-v2.json')
        folds = validate_membership(train, membership)
        counts = validate_counts(train, test, folds)
        list(cv_splits(train, folds))  # Fail before fitting if the split guard fails.
        tracking.params('data_counts', counts)
        tracking.params('input_sha256', input_before)
        tracking.params('source_config_sha256', before)
        tracking.write_json('inputs-manifest.json', dict(before=input_before, official_hashes_match=True))
        for relative in before:
            target = tracking.path('snapshot/' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            if sha256(target) != before[relative]:
                raise ValueError('source changed while snapshotting')
        report = execute_pipeline(tracking, train, test, folds, sample_ids)
        tracking.event('verify_immutable_inputs_and_source')
        input_after = data_manifest(Path(data_dir))
        after = source_manifest(ROOT)
        assert_unchanged(input_before, input_after, 'inputs')
        assert_unchanged(before, after, 'source/config/lock')
        tracking.write_json('immutability.json', dict(source_before=before, source_after=after,
                             input_before=input_before, input_after=input_after, all_unchanged=True,
                             closure_note='Only consumed runtime/config/dependencies. Presentation, docs, tools, evidence and outputs intentionally excluded.'))
        tracking.finish()
        print(json.dumps(dict(run_id=tracking.run_id, status=tracking.status, metrics=report['metrics'],
                              output_namespace=tracking.run_id, submitted=False)), flush=True)
        return report
