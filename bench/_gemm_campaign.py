"""Audit closed request campaigns without pooling timings across campaigns.

Usage: python3 bench/_gemm_campaign.py RUN_DIR [RUN_DIR ...]
Writes campaign-audit.json in each directory. Input JSON and logs stay intact.
"""
import hashlib
import json
from pathlib import Path
import random
import re
import statistics as st
import sys


def ci_ratio(control, candidate, paired=False):
    rng = random.Random(177)
    ratios = []
    for _ in range(4000):
        if paired:
            indices = rng.choices(range(len(control)), k=len(control))
            a = [control[i] for i in indices]
            b = [candidate[i] for i in indices]
        else:
            a = rng.choices(control, k=len(control))
            b = rng.choices(candidate, k=len(candidate))
        ratios.append(st.median(a) / st.median(b))
    ratios.sort()
    return [ratios[100], ratios[3899]]


def divergence(reference, candidate):
    first = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b), None)
    if first is None and len(reference) != len(candidate):
        first = min(len(reference), len(candidate))
    return dict(identical=reference == candidate, first_difference=first,
                reference_length=len(reference), candidate_length=len(candidate))


def audit(root):
    capture = root / 'capture.json'
    if capture.exists() and json.loads(capture.read_text()).get('diagnostic_only'):
        raise ValueError(f'{root}: Nsight diagnostic timings cannot enter a request campaign')
    groups = {}
    evidence = []
    for path in sorted(root.glob('*.json')):
        match = re.fullmatch(r'(herbert|vllm)-(0\.8B|4B)-([\w-]+?)(?:-r(\d+))?\.json', path.name)
        if not match:
            continue
        engine, model, variant, process = match.groups()
        data = json.loads(path.read_text())
        if data.get('timing') != 'token_ids_to_output_ids_greedy_ignore_eos' or not data.get('request_initialization_included'):
            raise ValueError(f'{path}: incompatible timing/reset scope')
        cases_path = root / f'cases-{model}.json'
        if not cases_path.exists():
            cases_path = root / 'cases.json'
        cases = json.loads(cases_path.read_text())
        case_lengths = {c['name']: len(c['ids']) for c in cases}
        if len(case_lengths) != len(cases) or not all(case_lengths.values()):
            raise ValueError(f'{cases_path}: duplicate or empty cases')
        output_counts = {s['output_tokens'] for s in data['samples']}
        expected = {(name, count, trial) for name in case_lengths
                    for count in output_counts for trial in range(data['rounds'])}
        actual = [(s['case'], s['output_tokens'], s['round']) for s in data['samples']]
        if not output_counts or len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{path}: missing, extra or duplicate case/output/round')
        evidence.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             cases_file=cases_path.name, cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                             context=data['context'], rounds=data['rounds'], warmup=data['warmup'],
                             tuning=data.get('resolved_tuning', data.get('tuning'))))
        for sample in data['samples']:
            if sample['prompt_tokens'] != case_lengths[sample['case']]:
                raise ValueError(f'{path}: input length differs from shared IDs')
            if len(sample['output_ids']) != sample['output_tokens']:
                raise ValueError(f'{path}: incomplete output IDs')
            key = model, engine, variant, sample['case'], sample['prompt_tokens'], sample['output_tokens']
            trial = int(process or 1), sample['round']
            bucket = groups.setdefault(key, {})
            if trial in bucket:
                raise ValueError(f'{path}: duplicate trial {key}/{trial}')
            bucket[trial] = sample
    if not groups:
        raise ValueError(f'{root}: no samples')
    if len({f['context'] for f in evidence}) != 1:
        raise ValueError(f'{root}: differing contexts')
    # A missing complete case must not disappear from the paired comparisons.
    for model in {k[0] for k in groups}:
        variants = {(k[1], k[2]) for k in groups if k[0] == model}
        grids = [{k[3:] for k in groups if k[:3] == (model, engine, variant)}
                 for engine, variant in variants]
        if any(grid != grids[0] for grid in grids):
            raise ValueError(f'{root}: differing case/output grids for {model}')
    result = dict(directory=str(root), evidence=evidence, rows=[], numerical=[])
    for key, samples in sorted(groups.items()):
        model, engine, variant, case, prompt, output = key
        sequence = next(iter(samples.values()))['output_ids']
        if any(s['output_ids'] != sequence for s in samples.values()):
            raise ValueError(f'{root}: nondeterministic IDs for {key}')
        times = [s['latency_seconds'] for s in samples.values()]
        row = dict(model=model, engine=engine, variant=variant, case=case, prompt=prompt,
                   output=output, n=len(times), median_ms=st.median(times)*1000,
                   min_ms=min(times)*1000, max_ms=max(times)*1000)
        if engine == 'herbert':
            ref_key = model, 'herbert', 'delivered', case, prompt, output
            reference = groups.get(ref_key)
            if reference is not None:
                if samples.keys() != reference.keys():
                    raise ValueError(f'{root}: unequal Herbert trial sets for {key}')
                trials = sorted(samples)
                ref_times = [reference[t]['latency_seconds'] for t in trials]
                trial_times = [samples[t]['latency_seconds'] for t in trials]
                row['ratio_delivered'] = st.median(ref_times) / st.median(trial_times)
                row['ci95_delivered'] = ci_ratio(ref_times, trial_times, paired=True)
            for ref_engine, ref_variant in [('herbert', 'auto'), ('herbert', 'delivered'), ('vllm', 'before'), ('vllm', 'after')]:
                ref = groups.get((model, ref_engine, ref_variant, case, prompt, output))
                if ref is not None:
                    ref_ids = next(iter(ref.values()))['output_ids']
                    result['numerical'].append(dict(model=model, variant=variant, case=case,
                        prompt=prompt, output=output, reference=f'{ref_engine}/{ref_variant}',
                        **divergence(ref_ids, sequence)))
            v_windows = []
            for window in ['before', 'after']:
                v = groups.get((model, 'vllm', window, case, prompt, output))
                if v is not None:
                    v_times = [s['latency_seconds'] for s in v.values()]
                    v_windows.extend(v_times)
                    row[f'ratio_vllm_{window}'] = st.median(v_times) / st.median(times)
                    row[f'ci95_vllm_{window}'] = ci_ratio(v_times, times)
            if v_windows:
                row['ratio_vllm_combined'] = st.median(v_windows) / st.median(times)
                row['ci95_vllm_combined'] = ci_ratio(v_windows, times)
                row['milliseconds_to_5pct'] = st.median(times)*1000 - st.median(v_windows)*1000/1.05
        result['rows'].append(row)
    result['vllm_drift'] = []
    for key, before in sorted(groups.items()):
        model, engine, variant, case, prompt, output = key
        if engine != 'vllm' or variant != 'before':
            continue
        after = groups.get((model, engine, 'after', case, prompt, output))
        if after is not None:
            result['vllm_drift'].append(dict(model=model, case=case, prompt=prompt, output=output,
                percent=100*(st.median(s['latency_seconds'] for s in after.values()) /
                             st.median(s['latency_seconds'] for s in before.values())-1)))
    (root / 'campaign-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(root, len(evidence), 'files,', len(result['rows']), 'groups,',
          sum(not x['identical'] for x in result['numerical']), 'numerical differences')


for directory in sys.argv[1:]:
    audit(Path(directory))
