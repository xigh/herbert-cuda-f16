"""Within-campaign Herbert gains, with paired bootstrap intervals by round."""
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
capture = root / 'capture.json'
if capture.exists() and json.loads(capture.read_text()).get('diagnostic_only'):
    raise ValueError(f'{root}: Nsight diagnostic timings cannot enter a request campaign')
values = defaultdict(dict)
for file in sorted(root.glob('herbert-*.json')):
    match = re.fullmatch(r'herbert-(0\.8B|4B)-([\w-]+?)(?:-r(\d+))?\.json', file.name)
    if not match: continue
    model, variant, process = match.groups()
    data = json.loads(file.read_text())
    if data.get('timing') != 'token_ids_to_output_ids_greedy_ignore_eos' or not data.get('request_initialization_included'):
        raise ValueError(f'{file}: incompatible timing/reset scope')
    for sample in data['samples']:
        key = model, variant, sample['prompt_tokens'], sample['output_tokens']
        trial = int(process or 1), sample['round']
        if trial in values[key]: raise ValueError(f'duplicate trial {key} {trial}')
        values[key][trial] = sample['latency_seconds']
control = sys.argv[2] if len(sys.argv) > 2 else 'delivered' if any(k[1] == 'delivered' for k in values) else 'auto'
print('model variant control prompt output paired_N median_control_ms median_variant_ms ratio CI95 verdict')
for key, samples in sorted(values.items()):
    model, variant, prompt, output = key
    if variant == control: continue
    ref = values.get((model, control, prompt, output), {})
    trials = sorted(samples.keys() & ref.keys())
    if not trials: continue
    pairs = [(ref[t], samples[t]) for t in trials]
    a, b = st.median(x[0] for x in pairs), st.median(x[1] for x in pairs)
    rng = random.Random(177)
    ratios = []
    for _ in range(4000):
        draw = rng.choices(pairs, k=len(pairs))
        ratios.append(st.median(x[0] for x in draw) / st.median(x[1] for x in draw))
    ratios.sort(); lo, hi = ratios[100], ratios[3899]
    verdict = 'GAIN' if lo > 1 else 'REGRESSION' if hi < 1 else 'UNCERTAIN'
    if len(pairs) < 20: verdict += '_N_LT_20'
    print(model, variant, control, prompt, output, len(pairs), f'{a*1000:.3f}', f'{b*1000:.3f}', f'{a/b:.5f}', f'{lo:.5f}..{hi:.5f}', verdict)
