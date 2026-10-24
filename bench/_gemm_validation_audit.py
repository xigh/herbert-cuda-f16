"""Audit tokens in the final numerical validation, without timing comparisons.

Usage: python3 bench/_gemm_validation_audit.py VALIDATION_DIR
CPU/GPU differences are preserved; candidate/delivered differences fail.
"""
import hashlib
import json
from pathlib import Path
import re
import sys


def difference(reference, candidate):
    first = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b), None)
    return dict(identical=reference == candidate, first_shared_token_difference=first,
                shared_prefix_identical=first is None, reference_length=len(reference),
                candidate_length=len(candidate))


def chat_ids(path):
    text = path.read_text()
    matches = re.findall(r'^\[ids\](.*)$', text, re.M)
    if len(matches) != 1 or not re.search(r'^\[turn ', text, re.M):
        raise ValueError(f'{path}: missing or ambiguous IDs/turn')
    ids = [int(x) for x in matches[0].split()]
    if not ids:
        raise ValueError(f'{path}: empty output IDs')
    return ids


root = Path(sys.argv[1])
expected_lengths = [63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1025, 2049, 4097]
prompts = ['p1-short', 'p2-explain', 'p3-long', 'p4-xlong', 'pp128', 'pp256', 'pp4000']
report = dict(scope='Numerical validation only; no latency ratios or timing confidence intervals',
              boundary_requests=0, models=[], oracle_comparisons=[], artifacts={})
for model in ['0.8B', '4B']:
    directory = root/model
    cases = json.loads((directory/'cases.json').read_text())
    if [len(c['ids']) for c in cases] != expected_lengths:
        raise ValueError(f'{directory}: differing boundary coverage')
    case_lengths = {c['name']: len(c['ids']) for c in cases}
    if len(case_lengths) != len(cases):
        raise ValueError(f'{directory}: duplicate cases')
    variants = {}
    for variant in ['auto', 'delivered', 'candidate']:
        path = directory/f'herbert-{model}-{variant}-r1.json'
        data = json.loads(path.read_text())
        if (data['rounds'], data['warmup'], data['context']) != (1, 1, 8192):
            raise ValueError(f'{path}: differing numerical protocol')
        if data['timing'] != 'token_ids_to_output_ids_greedy_ignore_eos' or not data['request_initialization_included']:
            raise ValueError(f'{path}: differing reset/scope')
        samples = {s['case']: s for s in data['samples']}
        if len(samples) != len(data['samples']) or samples.keys() != case_lengths.keys():
            raise ValueError(f'{path}: incomplete/duplicate boundary grid')
        for name, s in samples.items():
            if (s['round'], s['prompt_tokens'], s['output_tokens'], len(s['output_ids'])) != (0, case_lengths[name], 16, 16):
                raise ValueError(f'{path}: differing boundary sample')
        variants[variant] = samples
        report['boundary_requests'] += len(samples)
        report['artifacts'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for variant in ['delivered', 'candidate']:
        for name in case_lengths:
            d = difference(variants['auto'][name]['output_ids'], variants[variant][name]['output_ids'])
            rows.append(dict(case=name, prompt_tokens=case_lengths[name], variant=variant, **d))
            if not d['identical']:
                raise ValueError(f'{model}/{name}/{variant}: differs from auto1')
    report['models'].append(dict(model=model, boundaries=rows))
    oracle = directory/'oracle'
    for prompt in prompts:
        cpu_path = oracle/f'{prompt}-cpu.log'
        cpu = chat_ids(cpu_path)
        outputs = {}
        for variant in ['delivered', 'candidate']:
            path = oracle/f'{prompt}-cuda-0-{variant}.log'
            outputs[variant] = chat_ids(path)
            report['oracle_comparisons'].append(dict(model=model, prompt=prompt,
                variant=variant, reference='CPU oracle', **difference(cpu, outputs[variant])))
            report['artifacts'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['artifacts'][str(cpu_path.relative_to(root))] = hashlib.sha256(cpu_path.read_bytes()).hexdigest()
        if outputs['candidate'] != outputs['delivered']:
            raise ValueError(f'{model}/{prompt}: candidate differs from delivered chat')
report['note'] = 'CPU chat fixture max-tokens=2 preserves the first ID; lengths and all shared-token differences remain explicit. Existing CPU/GPU differences do not imply a new candidate defect.'
(root/'numerical-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(report['boundary_requests'], 'boundary requests; delivered/candidate IDs identical to auto1')
for r in report['oracle_comparisons']:
    if r['variant'] == 'delivered':
        print(r['model'], r['prompt'], 'CPU/GPU shared difference', r['first_shared_token_difference'],
              'lengths', r['reference_length'], r['candidate_length'])
