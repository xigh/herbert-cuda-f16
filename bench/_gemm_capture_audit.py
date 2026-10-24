"""Audit completed Nsight artifacts and actual launch geometry.

Usage: python3 bench/_gemm_capture_audit.py COUNTERS_DIR EXPECTED_CAPTURES
This checks preserved evidence; it does not benchmark instrumented requests.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from _ncu import load_capture


root = Path(sys.argv[1])
expected_count = int(sys.argv[2])
paths = sorted(root.glob('*/metrics.csv'))
if len(paths) != expected_count:
    raise ValueError(f'{root}: {len(paths)} captures, expected {expected_count}')
bench_sha = '38a8fc338c91354d226ea738b9097990d64f7e952c4d043e6f59784e617fed00'
required = ['capture.json', 'command.txt', 'binary-tool.sha256', 'nvidia-params.txt',
            'ncu-version.txt', 'ncu.log', 'counters.ncu-rep', 'metrics.csv',
            'details.txt', 'driver-sass.txt', 'source.csv', 'clocks-start.csv',
            'clocks-end.csv']
captures = []
for path in paths:
    directory = path.parent
    for name in required:
        if not (directory/name).is_file() or not (directory/name).stat().st_size:
            raise ValueError(f'{directory}: missing or empty {name}')
    c = load_capture(path)
    meta = c['metadata']
    cases_path = directory/'cases.json'
    if not cases_path.exists():
        cases_path = directory/'cases-4B.json'
    cases = json.loads(cases_path.read_text())
    selected = [case for case in cases if case['name'] == meta['case']]
    if len(selected) != 1 or len(selected[0]['ids']) != meta['prompt_tokens']:
        raise ValueError(f'{directory}: metadata differs from actual input IDs')
    if meta['first_block_m'] != min(meta['prompt_tokens'], meta['block']):
        raise ValueError(f'{directory}: differing first block size')
    if 'predecessor_prompt_tokens' in meta and [len(case['ids']) for case in cases] != [meta['predecessor_prompt_tokens'], meta['prompt_tokens']]:
        raise ValueError(f'{directory}: differing predecessor/input order')
    if not meta['diagnostic_only'] or meta['launch_count'] != 9:
        raise ValueError(f'{directory}: invalid diagnostic scope')
    if meta['clock_control'] != 'none' or meta['requested_sm_clock_mhz'] != 2805:
        raise ValueError(f'{directory}: differing clock protocol')
    if (meta['cache_control'], meta['replay_mode']) not in [('all', 'kernel'), ('none', 'application')]:
        raise ValueError(f'{directory}: unexpected replay/cache protocol')
    if not re.search(r'^RmProfilingAdminOnly: 0$', (directory/'nvidia-params.txt').read_text(), re.M):
        raise ValueError(f'{directory}: counter permission not captured')
    if bench_sha not in (directory/'binary-tool.sha256').read_text():
        raise ValueError(f'{directory}: differing measured binary')
    command = (directory/'command.txt').read_text()
    for key, value in [('clock-control', meta['clock_control']), ('cache-control', meta['cache_control']),
                       ('replay-mode', meta['replay_mode']), ('launch-skip', meta['launch_skip']),
                       ('launch-count', meta['launch_count'])]:
        if f'--{key} {value}' not in command:
            raise ValueError(f'{directory}: command/metadata differ for {key}')
    clocks = list(csv.DictReader((directory/'clocks-start.csv').open(), skipinitialspace=True))
    if int(clocks[0]['clocks.current.sm [MHz]'].split()[0]) != 2805:
        raise ValueError(f'{directory}: preflight clock differs')
    source = collections.Counter(row[1] for row in csv.reader((directory/'source.csv').open())
                                 if row and row[0] == 'Kernel Name')
    if source != collections.Counter({k['kernel']: k['launches'] for k in c['kernels']}):
        raise ValueError(f'{directory}: SASS/metrics occurrence sets differ')
    geometry = []
    for kernel in c['kernels']:
        match = re.fullmatch(r'gemm_(\d+)x(\d+)_(bf16|res)_(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)_s(\d+)(_xor)?(?:_sched\d+)?(?:_pipe\d+)?', kernel['kernel'])
        if match is None or kernel['launches'] != 3:
            raise ValueError(f'{directory}: unexpected kernel/occurrence count')
        kd, n, out, bm, bn, bk, wm, wn, stages, xor = match.groups()
        kd, n, bm, bn, bk, wm, wn, stages = map(int, (kd, n, bm, bn, bk, wm, wn, stages))
        actual_grid = tuple(map(int, re.findall(r'\d+', kernel['grid'])))
        actual_block = tuple(map(int, re.findall(r'\d+', kernel['block'])))
        expected_grid = (math.ceil(meta['first_block_m']/bm), n//bn, 1)
        if n % bn or actual_grid != expected_grid or actual_block != (32*wm*wn, 1, 1):
            raise ValueError(f'{directory}: actual geometry differs: {kernel["kernel"]} {actual_grid}')
        expected_shared = stages*(bm+bn)*(bk+(0 if xor else 8))*2
        if kernel['kernel'].endswith('_pipe3'):
            expected_shared += stages*16
        shared = kernel['metrics']['launch__shared_mem_per_block_dynamic']
        if shared['n'] != 3 or shared['min'] != expected_shared or shared['max'] != expected_shared:
            raise ValueError(f'{directory}: differing dynamic shared memory')
        role = 'gate_up' if (kd, n, out) == (2560, 18432, 'bf16') else 'down' if (kd, n, out) == (9216, 2560, 'res') else 'dn_projection' if (kd, n, out) == (2560, 12416, 'bf16') else None
        if role is None:
            raise ValueError(f'{directory}: unexpected matrix/output')
        geometry.append(dict(role=role, kernel=kernel['kernel'], grid=actual_grid,
                             grid_product=math.prod(actual_grid), block=actual_block,
                             dynamic_shared_bytes=expected_shared,
                             occurrences=kernel['launches']))
    if sorted(k['role'] for k in geometry) != ['dn_projection', 'down', 'gate_up']:
        raise ValueError(f'{directory}: missing matrix role')
    captures.append(dict(capture=directory.name, metadata=meta, geometry=geometry,
                         artifacts={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                    for name in [*required, cases_path.name]}))
result = dict(scope='Artifact, protocol and actual geometry audit; no request latency comparison',
              captures=len(captures), launches=sum(k['occurrences'] for c in captures for k in c['geometry']),
              evidence=captures)
(root/'capture-audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(result['captures'], 'closed captures,', result['launches'], 'actual launches; protocol/artifacts/geometry verified')
