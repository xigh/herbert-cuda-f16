"""Medians, dispersion and bootstrap confidence intervals of shared-ID runs."""
import collections
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
capture = root / 'capture.json'
if capture.exists() and json.loads(capture.read_text()).get('diagnostic_only'):
    raise SystemExit('Nsight diagnostic timings cannot enter the vLLM request comparison')
# Optional baseline files contribute IDs only, never timings from another run.
reference = Path(sys.argv[2]) if len(sys.argv) > 2 else None
rows = collections.defaultdict(list)
ids = {}
scopes = set()
files = list(root.glob('*.json')) + (list(reference.glob('*.json')) if reference else [])
for file in sorted(files):
    match = re.match(r'(herbert|vllm)-(0\.8B|4B)-([\w-]+?)(?:-r\d+)?\.json$', file.name)
    if not match:
        continue
    data = json.loads(file.read_text())
    if data.get('timing') != 'token_ids_to_output_ids_greedy_ignore_eos':
        raise SystemExit(f'{file}: unsupported timing scope')
    if file.parent == root:
        scopes.add(data.get('request_initialization_included', False))
    engine, model, variant = match.groups()
    variant = variant if engine == 'herbert' else 'vllm'
    for s in data['samples']:
        key = model, variant, s['prompt_tokens'], s['output_tokens']
        if file.parent == root:
            rows[key].append(s['latency_seconds'])
        token_key = key
        if token_key in ids and ids[token_key] != s['output_ids']:
            raise SystemExit(f'{key}: nondeterministic output IDs')
        if len(s['output_ids']) != s['output_tokens']:
            raise SystemExit(f'{file}: missing output IDs')
        ids[token_key] = s['output_ids']

if len(scopes) > 1:
    raise SystemExit('incompatible timing scopes: request initialization differs')
if scopes == {False}:
    print('WARNING: exploratory scope excludes Herbert request initialization')
if not rows:
    raise SystemExit('no request samples')

def interval(h, v):
    rng = random.Random(17)
    values = sorted(st.median(rng.choices(v, k=len(v))) / st.median(rng.choices(h, k=len(h))) for _ in range(4000))
    return values[100], values[3899]

print('model variant prompt output N median_ms range_ms ratio_vllm CI95 verdict')
for key, values in sorted(rows.items()):
    model, variant, prompt, output = key
    median = st.median(values)
    v = rows.get((model, 'vllm', prompt, output))
    ratio = st.median(v) / median if v else None
    ci = interval(values, v) if v and variant != 'vllm' else None
    verdict = '-'
    if ci:
        verdict = 'EXPLORATORY' if min(len(values), len(v)) < 20 else 'WIN_5PCT' if ratio >= 1.05 and ci[0] > 1 else 'AHEAD' if ci[0] > 1 else 'BEHIND' if ci[1] < 1 else 'UNCERTAIN'
    print(model, variant, prompt, output, len(values), f'{median*1000:.3f}',
          f'{min(values)*1000:.3f}..{max(values)*1000:.3f}',
          f'{ratio:.4f}' if ratio else '-', f'{ci[0]:.4f}..{ci[1]:.4f}' if ci else '-', verdict)

print('\nHerbert candidate vs auto output IDs:')
for key, tokens in sorted(ids.items()):
    model, variant, prompt, output = key
    if variant in ('vllm', 'auto'):
        continue
    ref = ids.get((model, 'auto', prompt, output))
    if ref is not None:
        first = next((i for i, (x, y) in enumerate(zip(ref, tokens)) if x != y), None)
        print(model, variant, prompt, output, 'identical' if ref == tokens else f'first divergence {first}')

print('\nHerbert vs vLLM output IDs (different numerical implementations):')
for key, tokens in sorted(ids.items()):
    model, variant, prompt, output = key
    if variant == 'vllm': continue
    ref = ids.get((model, 'vllm', prompt, output))
    if ref is not None:
        first = next((i for i, (x, y) in enumerate(zip(ref, tokens)) if x != y), None)
        print(model, variant, prompt, output, 'identical' if ref == tokens else f'first divergence {first}')

print('\nvLLM before/after median drift:')
for before in sorted(root.glob('vllm-*-before.json')):
    after = root / before.name.replace('-before', '-after')
    if not after.exists(): continue
    windows = []
    for f in [before, after]:
        d = collections.defaultdict(list)
        for sample in json.loads(f.read_text())['samples']:
            d[sample['prompt_tokens'], sample['output_tokens']].append(sample['latency_seconds'])
        windows.append(d)
    for key, v in sorted(windows[0].items()):
        print(before.name.split('-')[1], *key, f'{(st.median(windows[1][key])/st.median(v)-1)*100:+.2f}%')
