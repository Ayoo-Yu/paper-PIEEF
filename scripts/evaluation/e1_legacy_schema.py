"""Read-only, hash-bound adapter for exactly one previously produced Huber run.

No model loading, inference, metadata mutation, or fallback split default occurs here.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid

CONTRACT_SHA256 = "80b732c24f3a21e435da5511ac132a91e9770f4f9ad6634588365b9847926b5d"


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _need(condition, message):
    if not condition:
        raise ValueError('Legacy Huber source refused: ' + message)


def _checked_file(path, expected):
    path = Path(path)
    _need(path.is_file(), 'missing evidence ' + str(path))
    _need(path.stat().st_size == expected['bytes'] and _sha(path) == expected['sha256'],
          'evidence SHA/size mismatch ' + str(path))


def _relative_legacy_path(value):
    text = str(value).replace('\\', '/')
    marker = '/补做重算/'
    _need(text.count(marker) == 1, 'unrecognised legacy path')
    return '补做重算/' + text.split(marker, 1)[1]


def _equal_test_arrays(left, right, label):
    import numpy as np
    for key in ('timestamp', 'origin_timestamp', 'turbine_ids', 'target_valid', 'events'):
        a, b = left[key], right[key]
        if a.dtype.kind == 'O':
            a = a.astype(str)
        if b.dtype.kind == 'O':
            b = b.astype(str)
        _need(np.array_equal(a, b), label + ': unequal ' + key)
    _need(np.array_equal(left['y_true_mw'], right['y_true_mw'], equal_nan=True),
          label + ': unequal observed targets')


def _audit(args, record):
    home = Path(args.output_root).resolve() / 'ops' / 'legacy_schema_audits'
    home.mkdir(parents=True, exist_ok=True)
    record.update(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=os.getpid())
    path = home / ('source_schema_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
                   + '_' + uuid.uuid4().hex + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def verify_test_source(args, task, source, meta):
    """Return a separately verified split; never insert a field in source metadata."""
    record = dict(adapter='exact-huber-h16-seed13-legacy-schema-v1', status='FAIL',
                  source=str(source), task_id=task.get('task_id'), source_metadata_modified=False,
                  model_loaded=False, inference_executed=False)
    try:
        contract_path = Path(__file__).with_name('e1_huber_legacy_contract.json')
        _need(_sha(contract_path) == CONTRACT_SHA256, 'contract changed')
        contract = _json(contract_path)
        record['contract_sha256'] = CONTRACT_SHA256
        _need('evaluation_split' not in meta, 'adapter is only for an absent split field')
        _need(task.get('verification_kind') == 'huber' and task.get('model') == 'ACFormer'
              and task.get('seed') == 13 and task.get('horizon') == 16
              and task.get('task_id') == 'ACFormer_huber_h16_seed13' and args.horizon == 16,
              'task not in the single-source whitelist')
        reference_root, source = Path(args.reference_root).resolve(), Path(source).resolve()
        expected_source = (reference_root / contract['source_relative']).resolve()
        _need(expected_source.is_relative_to(reference_root) and source == expected_source,
              'source is not the exact declared attempt')
        _need(source.name == contract['attempt_id'] and source.parent.name == contract['config_id'],
              'attempt/config directory differs')
        _need((meta['status'], meta['model'], meta['seed'], meta['horizon'], meta['run_id']) ==
              ('completed', 'ACFormer', 13, 16, contract['config_id']), 'source metrics identity differs')
        cfg = meta['config']
        _need(cfg['tracking']['config_id'] == contract['config_id']
              and cfg['experiment']['horizon'] == 16 and cfg['train']['loss_name'] == 'huber'
              and cfg['train']['huber_delta'] == .1, 'configuration differs')
        for name, expected in contract['source_files'].items():
            _checked_file(source / name, expected)
        # The caller must supply the exact disk object, not a fabricated in-memory schema.
        _need(meta == _json(source / 'metrics.json'), 'in-memory metrics differs from source')
        execution = _json(source / 'execution.json')
        _need(execution['status'] == 'completed' and execution['returncode'] == 0
              and execution['config_id'] == contract['config_id'], 'source did not exit successfully')
        refs = {}
        for name, expected in contract['reference_files'].items():
            path = (reference_root / expected['relative']).resolve()
            _need(path.is_relative_to(reference_root), 'reference escapes its root')
            _checked_file(path, expected)
            refs[name] = path
        selected = [x for x in _json(refs['selected']) if
                    (x['model'], x['horizon'], x['seed']) == ('ACFormer', 16, 13)]
        _need(len(selected) == 1 and _relative_legacy_path(selected[0]['directory']) == contract['source_relative'],
              'selected source binding differs')
        f2 = _json(refs['f2_completed'])
        _need(f2['status'] == 'completed' and f2['trained'] is False
              and (f2['model'], f2['horizon'], f2['seed']) == ('ACFormer', 16, 13)
              and _relative_legacy_path(f2['source_directory']) == contract['source_relative'],
              'F2 completion identity differs')
        checkpoint_sha = _sha(source / 'weights_best.pt')
        baseline_sha = _sha(refs['baseline_manifest'])
        _need(checkpoint_sha == contract['checkpoint_sha256'], 'wrong checkpoint')
        expected_f2_source = dict(config_id=contract['config_id'], run_id=contract['attempt_id'],
                                  checkpoint_sha256=checkpoint_sha, source_manifest_sha256=baseline_sha)
        _need(f2['source'] == expected_f2_source, 'F2 checkpoint/manifest binding differs')
        baseline_sources = [x for x in _json(refs['baseline_manifest'])['sources']
                            if _relative_legacy_path(x['path']) == contract['source_relative']]
        _need(len(baseline_sources) == 1, 'missing unique F2 source manifest entry')
        source_entry = baseline_sources[0]
        _need((source_entry['model'], source_entry['horizon'], source_entry['seed'], source_entry['config_id'])
              == ('ACFormer', 16, 13, contract['config_id']), 'F2 source record identity differs')
        members = {Path(x['path'].replace('\\', '/')).name: x for x in source_entry['files']}
        _need(set(members) == set(contract['source_files']), 'F2 source member set differs')
        for name, expected in contract['source_files'].items():
            _need(_relative_legacy_path(members[name]['path']) == contract['source_relative'] + '/' + name
                  and members[name]['sha256'] == expected['sha256'], 'F2 source member binding differs')
        # Frozen producer proof: original package/runner and save_run_predictions implementation.
        package = Path(__file__).resolve().parents[1]
        proof = {}
        for name, expected in contract['producer_evidence'].items():
            path = (package / expected['relative']).resolve()
            _need(path.is_relative_to(package), 'producer proof escapes package')
            _checked_file(path, expected)
            proof[name] = path
        provenance = _json(source / 'provenance.json')
        _need(provenance['package_manifest_sha256'] == _sha(proof['package_manifest'])
              and provenance['source_manifest_sha256'] == _sha(source / 'source_manifest.json')
              and meta['source_manifest_sha256'] == _sha(source / 'source_manifest.json'),
              'original producer provenance differs')
        package_entries = {x['path']: x for x in _json(proof['package_manifest'])['files']}
        _need(package_entries['runner.py']['sha256'] == _sha(proof['runner']), 'frozen runner mismatch')
        source_entries = {x['path']: x for x in _json(source / 'source_manifest.json')['files']}
        _need(source_entries['candidate_project/src/pieef/artifacts.py']['sha256'] == _sha(proof['artifacts']),
              'frozen prediction writer mismatch')
        import numpy as np
        with np.load(source / 'predictions.npz', allow_pickle=False) as saved, \
             np.load(source / 'preprocessing.npz', allow_pickle=False) as pre, \
             np.load(refs['canonical_normal'], allow_pickle=False) as canonical, \
             np.load(refs['f2_normal'], allow_pickle=False) as normal:
            _equal_test_arrays(saved, canonical, 'source/canonical test')
            _equal_test_arrays(saved, normal, 'source/F2 test')
            _need(int(saved['seed']) == 13 and int(saved['horizon']) == 16
                  and saved['y_true_mw'].shape == (7604, 29)
                  and saved['target_valid'].dtype.kind == 'b'
                  and int(saved['target_valid'].sum()) == 220401
                  and np.array_equal(np.isfinite(saved['y_true_mw']), saved['target_valid']),
                  'source test support differs')
            _need(np.array_equal(pre['capacity_mw'], canonical['capacity_mw'])
                  and np.array_equal(pre['capacity_mw'], normal['capacity_mw']), 'capacity differs')
        for name, expected in contract['f2_predictions'].items():
            path = refs['f2_normal'].parent / name
            _checked_file(path, expected)
            with np.load(path, allow_pickle=False) as z:
                for key, value in dict(seed=13, horizon=16, fault_seed=2026, scenario=Path(name).stem,
                                       source_config_id=contract['config_id'], source_run_id=contract['attempt_id'],
                                       checkpoint_sha256=checkpoint_sha, source_manifest_sha256=baseline_sha).items():
                    _need(z[key].item() == value, 'F2 ' + name + ' binding differs: ' + key)
        # Check all original source bytes again after evidence reads, before returning control.
        for name, expected in contract['source_files'].items():
            _checked_file(source / name, expected)
        record.update(status='PASS', verified_evaluation_split='test', source_file_count=16,
                      f2_prediction_count=40, target_shape=[7604, 29], valid_targets=220401,
                      checkpoint_sha256=checkpoint_sha, baseline_manifest_sha256=baseline_sha,
                      basis='Hash-bound producer test output mapping plus exact canonical/F2 test arrays')
        _audit(args, record)
        return 'test'
    except Exception as error:
        record['error'] = str(error)
        _audit(args, record)
        raise
